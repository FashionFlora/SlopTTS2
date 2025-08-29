# filename: tts_save_tensors.py
# Purpose: same pipeline but extended to save intermediate tensors/vars to output.txt
# Ensure you run this in the same environment where your models and utils are available.

import os
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import soundfile as sf
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import time
import yaml
from munch import Munch
import torchaudio
import librosa
from scipy import signal
from collections import OrderedDict

from models import *
from utils import *
from text_utils import TextCleaner

textclenaer = TextCleaner()
device = "cpu"

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=128, n_fft=4096, win_length=1200, hop_length=600
)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(
        lengths.shape[0], -1
    ).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# helper to robustly convert to numpy and save readable output
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return str(x)

def save_tensor(f, name, tensor):
    """
    Save a variable to the opened file object f in a readable way.
    Accepts torch.Tensor, numpy.ndarray, Python scalars, lists, tuples, dicts.
    """
    f.write(f"\n=== {name} ===\n")
    try:
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
            f.write(f"Type: torch.Tensor\nShape: {tuple(tensor.shape)}\n")
            f.write(np.array2string(arr, threshold=10_000, max_line_width=200))
            f.write("\n")
        elif isinstance(tensor, np.ndarray):
            f.write("Type: np.ndarray\n")
            f.write(f"Shape: {tensor.shape}\n")
            f.write(np.array2string(tensor, threshold=10_000, max_line_width=200))
            f.write("\n")
        elif isinstance(tensor, (list, tuple)):
            f.write(f"Type: {type(tensor)} Length: {len(tensor)}\n")
            f.write(str(tensor) + "\n")
        elif isinstance(tensor, dict):
            f.write(f"Type: dict Length: {len(tensor)}\n")
            for k, v in tensor.items():
                f.write(f"  {k}: ")
                try:
                    f.write(str(_to_numpy(v)) + "\n")
                except Exception:
                    f.write(repr(v) + "\n")
        else:
            # scalar or other
            f.write(f"Type: {type(tensor)}\nValue: {tensor}\n")
    except Exception as e:
        f.write(f"Could not save variable due to exception: {repr(e)}\n")
# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="pl", preserve_punctuation=True, with_stress=True
)

config = yaml.safe_load(open("./Configs/config.yml"))

# load pretrained ASR, F0, BERT, model
ASR_config = config.get("ASR_config", False)
ASR_path = config.get("ASR_path", False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

F0_path = config.get("F0_path", False)
pitch_extractor = load_F0_models(F0_path)

from Utils.PLBERT.util import load_plbert

BERT_path = config.get("PLBERT_dir", False)
plbert = load_plbert(BERT_path)

model = build_model(recursive_munch(config["model_params"]), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load(
    "again_wavlm/epoch_2nd_00030.pth", map_location="cpu"
)
params = params_whole["net"]

for key in model:
    if key in params:
        print("%s loaded" % key)
        try:
            model[key].load_state_dict(params[key])
        except Exception:
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            model[key].load_state_dict(new_state_dict, strict=False)
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, KarrasSchedule,DPMpp2MSampler
'''
sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
'''
sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=DPMpp2MSampler(s_churn=0.0, s_tmin=0.5, s_tmax=3.0, s_noise=1.0),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )
def LFinference(
    text,
    s_prev,
    noise,
    alpha=0.7,
    diffusion_steps=5,
    embedding_scale=1,
    out_txt_path="output.txt",
    # NEW optional arguments:
    ref_wave=None,   # 1D numpy waveform (sr same as training) OR torch tensor [T]
    ref_mel=None,    # mel spectrogram tensor/ndarray [n_mels, T] already precomputed
    use_gt_style=True,  # if True and ref provided -> use GT style; otherwise use diffusion
):
    """
    Inference helper that saves intermediate tensors. New: can take style from a GT
    example (ref_wave or ref_mel) similarly to the training code:
      s_dur = model.predictor_encoder(gt_mel.unsqueeze(1))
      s_ss  = model.style_encoder(gt_mel.unsqueeze(1))
      ref = torch.cat([s_ss, s_dur], dim=1)

    Parameters:
    - ref_wave: numpy 1D waveform (float) sampled at training sr OR torch 1D Tensor
    - ref_mel: mel spectrogram (n_mels, T) as numpy or torch
    - use_gt_style: if True and ref provided -> use GT style; otherwise fallback to diffusion sampler
    """
    text = text.strip().replace('"', "")
    ps = global_phonemizer.phonemize([text])
    ps = ps[0].split()
    ps = " ".join(ps)
    tokens = textclenaer(ps)

    # keep same token additions as original
    tokens.insert(0, 0)
    tokens_tensor = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad(), open(out_txt_path, "w", encoding="utf-8") as f:
        input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(tokens_tensor.device)
        text_mask = length_to_mask(input_lengths).to(tokens_tensor.device)

        save_tensor(f, "tokens", tokens_tensor)
        save_tensor(f, "input_lengths", input_lengths)
        save_tensor(f, "text_mask", text_mask)

        t_en = model.text_encoder(tokens_tensor, input_lengths, text_mask)
        save_tensor(f, "t_en", t_en)

        bert_dur = model.bert(tokens_tensor, attention_mask=(~text_mask).int())
        save_tensor(f, "bert_dur", bert_dur)

        # unify bert_dur_first
        if isinstance(bert_dur, (list, tuple)):
            bert_dur_first = bert_dur[0]
        else:
            bert_dur_first = bert_dur
        save_tensor(f, "bert_dur_first", bert_dur_first)

        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        save_tensor(f, "d_en", d_en)

        # ---------- NEW: compute style FROM GT reference if provided ----------
        style_from_gt = False
        ref = None
        s_pred = None

        if use_gt_style and (ref_mel is not None or ref_wave is not None):
            # prepare mel tensor
            if ref_mel is None and ref_wave is not None:
                # expect ref_wave numpy 1D, use preprocess to produce normalized log-mel (same as preprocess)
                if isinstance(ref_wave, torch.Tensor):
                    ref_wave_tensor = ref_wave.float().cpu().numpy()
                else:
                    ref_wave_tensor = np.array(ref_wave, dtype=float)
                # compute mel with same mel transform used in preprocess
                ref_mel_t = to_mel(torch.from_numpy(ref_wave_tensor).float())
                ref_mel_t = (torch.log(1e-5 + ref_mel_t) - mean) / std
            else:
                # ref_mel provided
                if isinstance(ref_mel, np.ndarray):
                    ref_mel_t = torch.from_numpy(ref_mel).float()
                else:
                    ref_mel_t = ref_mel.clone().float()
                # If ref_mel lacks batch/channel dims, ensure shape [n_mels, T]
                # training expects inputs with shape [B, 1, T_mel, ...] for encoder; we add dims below.

            # ensure shape [1, 1, n_mels, T] or [1, n_mels, T] depending on encoder
            # In your training you call: ref_mels.unsqueeze(1) -> shape [B, 1, n_mels, T]
            if ref_mel_t.dim() == 2:
                ref_mel_t = ref_mel_t.unsqueeze(0)  # [1, n_mels, T]
            if ref_mel_t.dim() == 3:
                # [1, n_mels, T] -> add channel dim
                ref_mel_in = ref_mel_t.unsqueeze(1).to(device)  # [1,1,n_mels,T] if your encoders expect that
            else:
                # already has channel dim
                ref_mel_in = ref_mel_t.to(device)

            # compute style pieces same as training:
            try:
                ref_ss = model.style_encoder(ref_mel_in)        # global acoustic style
                ref_sp = model.predictor_encoder(ref_mel_in)    # global prosodic style
                ref = torch.cat([ref_ss, ref_sp], dim=1)        # [1, style_dim]
                style_from_gt = True
                save_tensor(f, "ref_mel_in", ref_mel_in)
                save_tensor(f, "ref_ss", ref_ss)
                save_tensor(f, "ref_sp", ref_sp)
                save_tensor(f, "ref_style_concat", ref)
            except Exception as e:
                # if style encoders fail, fall back to diffusion later
                save_tensor(f, "style_extraction_exception", repr(e))
                style_from_gt = False

        # ---------- diffusion fallback: compute s_pred via sampler as previously ----------
        if not style_from_gt:
            s_pred = sampler(
                noise,
                embedding=bert_dur_first[0].unsqueeze(0),
                num_steps=12,
                embedding_scale=1,
            ).squeeze(0)
            save_tensor(f, "s_pred", s_pred)

            if s_prev is not None:
                s_pred = alpha * s_prev + (1 - alpha) * s_pred
                save_tensor(f, "s_pred_after_mix", s_pred)

            # split ref and s (consistent with your previous code)
            if s_pred.dim() == 2 and s_pred.size(1) > 128:
                ref = s_pred[:, :128]
                s = s_pred[:, 128:]
            else:
                ref = s_pred[..., :128]
                s = s_pred[..., 128:]
            save_tensor(f, "ref", ref)
            save_tensor(f, "s", s)
        else:
            # we have ref (style vector) from GT. Training concatenates [ss, sp].
            # ref is already [1, style_dim]. Need to split 'ref' into the halves used before:
            # in training you did ref = torch.cat([ref_ss, ref_sp], dim=1)
            # In your inference code previously you expected ref = s_pred[:, :128] and s = s_pred[:, 128:]
            # We'll try to maintain the same split length if possible.
            ref = ref.to(device)
            # If style dim >= 256 we follow original split convention, otherwise attempt reasonable split in half.
            style_dim = ref.size(1)
            if style_dim >= 256:
                ref_slice = ref[:, :128]
                s = ref[:, 128:128 + (style_dim - 128)]
            else:
                # split half/half
                half = style_dim // 2
                ref_slice = ref[:, :half]
                s = ref[:, half:]
            save_tensor(f, "ref_from_gt", ref_slice)
            save_tensor(f, "s_from_gt", s)
            ref = ref_slice

        # continue with rest of pipeline
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        save_tensor(f, "d", d)

        x, _ = model.predictor.lstm(d)
        save_tensor(f, "x", x)

        duration_raw = model.predictor.duration_proj(x)
        save_tensor(f, "duration_raw", duration_raw)

        duration = torch.sigmoid(duration_raw).sum(axis=-1)
        save_tensor(f, "duration_sigmoid_sum", duration)

        pred_dur = torch.round(duration.squeeze()).clamp(min=1).to(torch.long)
        save_tensor(f, "pred_dur", pred_dur)

        # build pred_aln_trg robustly (handle shape mismatches)
        tokens_len = int(input_lengths.squeeze().item())
        pred_dur_arr = pred_dur.detach().cpu().numpy().astype(int)
        if pred_dur_arr.size == 1 and tokens_len > 1:
            pred_dur_arr = np.full((tokens_len,), int(pred_dur_arr.item()), dtype=int)
        if pred_dur_arr.size != tokens_len:
            if pred_dur_arr.size < tokens_len:
                pad = np.ones((tokens_len - pred_dur_arr.size,), dtype=int)
                pred_dur_arr = np.concatenate([pred_dur_arr, pad], axis=0)
            else:
                pred_dur_arr = pred_dur_arr[:tokens_len]

        total_frames = int(pred_dur_arr.sum()) if pred_dur_arr.sum() > 0 else 1
        pred_aln_trg = torch.zeros((tokens_len, total_frames), dtype=torch.float32)
        c_frame = 0
        for i in range(tokens_len):
            dur_i = int(pred_dur_arr[i])
            if dur_i <= 0:
                dur_i = 1
            end = min(c_frame + dur_i, total_frames)
            pred_aln_trg[i, c_frame:end] = 1
            c_frame += dur_i
            if c_frame >= total_frames:
                break

        save_tensor(f, "pred_aln_trg", pred_aln_trg)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        save_tensor(f, "en", en)

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        save_tensor(f, "F0_pred", F0_pred)
        save_tensor(f, "N_pred", N_pred)

        t_en_aligned = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        save_tensor(f, "t_en_aligned", t_en_aligned)

        ref_in = ref.squeeze().unsqueeze(0).to(device)
        out = model.decoder(t_en_aligned, F0_pred, N_pred, ref_in)
        save_tensor(f, "out", out)

    # return waveform and style vector for chaining
    return out.squeeze().cpu().numpy(), s_pred
# -------------------------
# Example usage (same as original)
# -------------------------
if __name__ == "__main__":
    passage = (
        "Jeśli korzystasz z laptopa dłużej niż godzinę dziennie, "
        "podstawka pod laptopa może znacznie zmniejszyć obciążenie szyi i pleców"
    )

    def smart_text_chunking(passage, target_length=150):
        raw_sentences = passage.split(".")
        sentences = [s.strip() + "." for s in raw_sentences if s.strip()]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= target_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    chunks = smart_text_chunking(passage, target_length=150)
    wavs = []
    s_prev = None
    N_ZEROS = 5000
    os.makedirs("./samples", exist_ok=True)

    ref_path = "test.wav"
    ref_wave, sr = librosa.load(ref_path, sr=None, mono=True)

    # if your training sr differs, resample:
    train_sr = 44100  # or read config['preprocess_params']['sr']
    if sr != train_sr:
        ref_wave = librosa.resample(ref_wave, orig_sr=sr, target_sr=train_sr)

    # call inference using GT style
    out_wav, used_style = LFinference(
        "Jeśli korzystasz z laptopa dłużej niż godzinę dziennie, to jest źle. Podstawka pod laptopa może znacznie zmniejszyć obciążenie szyi i pleców",
        s_prev=None,
        noise=torch.randn(1, 1, 256).to(device),
        ref_wave=ref_wave,
        use_gt_style=True,
        out_txt_path="out_with_gt_style.txt"
    )

    # save output
    import soundfile as sf
    sf.write("out_with_gt_style.wav", out_wav, samplerate=train_sr)