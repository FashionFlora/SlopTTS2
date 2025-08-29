import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor, WhisperModel, WhisperConfig, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import whisper

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram( n_mels=128, sample_rate=44100, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss

class MagPhaseLoss(torch.nn.Module):
  
    def __init__(self, *, n_fft, hop_length, eps=1e-10,
                 mag_weight=1.0, phase_weight=0.1):
        super().__init__()
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight

    def forward(self, y_rec_mag, y_rec_phase, gt):
    
        if y_rec_mag is None or y_rec_phase is None:
            raise ValueError("Predicted mag/phase must not be None")

   
        gt = gt.to(y_rec_mag.device, dtype=y_rec_mag.dtype)

  
        window = self.window
        if window.dtype != gt.dtype:
            window = window.to(dtype=gt.dtype)

 
        y_stft = torch.stft(
            gt,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )

        target_mag = torch.abs(y_stft).clamp(min=self.eps) 
        target_phase = torch.angle(y_stft)  # in radians, [-pi, pi]


        # magnitude loss (L1)
        loss_mag = F.l1_loss(y_rec_mag, target_mag)

        # phase loss: compare sin/cos to avoid wrap-around issues
        loss_phase = F.l1_loss(torch.cos(y_rec_phase),
                               torch.cos(target_phase)) \
                     + F.l1_loss(torch.sin(y_rec_phase),
                                 torch.sin(target_phase))

        loss = self.mag_weight * loss_mag + self.phase_weight * loss_phase
        return loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 # NEW, CORRECTED PARAMETERS FOR 44.1kHz
                 fft_sizes=[2048, 4096, 1024],
                 hop_sizes=[240, 480, 160],
                 win_lengths=[1200, 2400, 800],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(GeneratorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)


        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss.mean()
   
class WhisperEncoderOnly(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)

    def forward(self, input_features, attention_mask=None, output_hidden_states=None):
        # you may want to forward through encoder and return the ModelOutput-like object
        return self.encoder(input_features, attention_mask=attention_mask,
                            output_hidden_states=output_hidden_states)

class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()

        # Load the full whisper-small model and extract only the encoder
        full_model = WhisperModel.from_pretrained(
            "openai/whisper-small",
            device_map='auto',           # OK if you have accelerate support; otherwise omit
            torch_dtype=torch.bfloat16   # change if your GPU doesn't support bfloat16
        )

        # Create encoder-only model with the same config and copy weights
        config = full_model.config
        encoder_only = WhisperEncoderOnly(config)
        encoder_only.encoder.load_state_dict(full_model.encoder.state_dict())
        del full_model

        self.wavlm = encoder_only.to(torch.bfloat16)  # or torch.float16 / torch.float32
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        for p in self.wavlm.parameters():
            p.requires_grad = False

    def forward(self, wav, y_rec, generator=False, discriminator=False, discriminator_forward=False):
        wav = wav.squeeze(1)
        y_rec = y_rec.squeeze(1)
        wav = whisper.pad_or_trim(wav)
        wav = whisper.log_mel_spectrogram(wav)

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec)

        with torch.no_grad():
            wav_embeddings = self.wavlm.encoder(
                wav.to(torch.bfloat16),
                output_hidden_states=True
            ).hidden_states
        y_rec_embeddings = self.wavlm.encoder(
            y_rec.to(torch.bfloat16),
            output_hidden_states=True
        ).hidden_states

        pooled_weight = 1.0
        frame_weight = 0.2  
        layer_weights = None 
        loss =0
        total = 0.0
        n = len(wav_embeddings)
        if layer_weights is None:
            layer_weights = [1.0] * n

        for w, er, eg in zip(layer_weights, wav_embeddings, y_rec_embeddings):
            er = er.float()  
            eg = eg.float()

            er_pool = er.mean(dim=1)   
            eg_pool = eg.mean(dim=1)
            pooled_l1_per = torch.mean(torch.abs(er_pool - eg_pool), dim=1) 
            frame_l1_per  = torch.mean(torch.abs(er - eg), dim=(1, 2))

            layer_loss = pooled_weight * pooled_l1_per + frame_weight * frame_l1_per
            #layer_loss = frame_l1_per
            total += w * layer_loss

        
        return total.mean()
    def generator(self, y_rec):
        
        y_rec = y_rec.squeeze(1)
        

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec)

        with torch.no_grad():
            y_rec_embeddings = self.wavlm.encoder(y_rec.to(torch.bfloat16), output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings.to(torch.float32))
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen.to(torch.float32)

    def discriminator(self, wav, y_rec):
        
        wav = wav.squeeze(1)
        y_rec = y_rec.squeeze(1)

        wav = whisper.pad_or_trim(wav)
        wav = whisper.log_mel_spectrogram(wav)

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec)

        with torch.no_grad():
            wav_embeddings = self.wavlm.encoder(wav.to(torch.bfloat16), output_hidden_states=True).hidden_states
            y_rec_embeddings = self.wavlm.encoder(y_rec.to(torch.bfloat16), output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings.to(torch.float32))
        y_d_gs = self.wd(y_rec_embeddings.to(torch.float32))
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f.mean().to(torch.float32)
    
    


    def discriminator_forward(self, wav):
        # Squeeze the channel dimension if it's unnecessary
        wav = wav.squeeze(1) # Adjust this line if the channel dimension is not at dim=1


        with torch.no_grad():
            
            wav_16 = self.resample(wav)
            wav_16 = whisper.pad_or_trim(wav_16)
            wav_16 = whisper.log_mel_spectrogram(wav_16)
            
            wav_embeddings = self.wavlm.encoder(wav_16.to(torch.bfloat16) , output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings.to(torch.float32))
        
        return y_d_rs