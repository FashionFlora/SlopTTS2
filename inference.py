# load packages
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
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()


device = 'cpu'


# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='pl', preserve_punctuation=True, with_stress=True , words_mismatch='ignore')

config = yaml.safe_load(open("./Configs/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("again_wavlm/epoch_2nd_00006.pth", map_location='cpu')
params = params_whole['net']


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


from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule,DPMpp2MSampler

sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

def check_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}!")
    else:
        print(f"✅ {name} OK (no NaNs)")

def LFinference(text, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        # 1. Text + BERT embeddings
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        check_nan("t_en", t_en)

        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        check_nan("bert_dur", bert_dur)

        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        check_nan("d_en", d_en)

        # 2. Diffusion style vector
        s_pred = sampler(
            noise,
            embedding=bert_dur[0].unsqueeze(0),
            num_steps=6,
            embedding_scale=embedding_scale,
        ).squeeze(0)
        print(s_pred.shape)
        check_nan("s_pred", s_pred)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        check_nan("s", s)
        check_nan("ref", ref)

        # 3. Duration prediction
        d_full = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        check_nan("d_full", d_full)

        pred_dur = model.predictor.sample_durations(
            d_full=d_full,
            texts=d_en,
            style=s,
            text_lengths=input_lengths,
            m=text_mask,
            steps=32,
        ).squeeze(0)
        check_nan("pred_dur", pred_dur)
        print(pred_dur)
        # 4. Alignment matrix
        pred_aln_trg = torch.zeros(
            input_lengths, int(pred_dur.sum().data), device=device
        )
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
        check_nan("pred_aln_trg", pred_aln_trg)

        # 5. Cross-attention enriched features
        en = model.predictor.compute_en(
            d_tok=d_full,
            alignment=pred_aln_trg.unsqueeze(0),
            style=s,
            text_mask=text_mask,
        )
        check_nan("en", en)

        # 6. F0 + Norm prediction
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        check_nan("F0_pred", F0_pred)
        check_nan("N_pred", N_pred)

        # 7. Decoder output
        out = model.decoder(
            (t_en @ pred_aln_trg.unsqueeze(0)), F0_pred, N_pred, ref.squeeze().unsqueeze(0)
        )
        check_nan("out", out)

    return out.squeeze().cpu().numpy(), s_pred
passage = '''„Silmarillion”, wydany w cztery lata po śmierci autora, jest relacją o Dawnych Dniach'''

def smart_text_chunking(passage, target_length=100):
    # First, properly split the passage into sentences
    raw_sentences = passage.split('.')
    sentences = [s.strip() + '.' for s in raw_sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would keep us under target length, add it to current chunk
        if len(current_chunk) + len(sentence) <= target_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            # If current chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
            # Start a new chunk with the current sentence
            current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Use the improved chunking in the TTS pipeline
chunks = smart_text_chunking(passage, target_length=150)  # Adjust target length as needed
wavs = []
s_prev = None



def create_smooth_transition(audio1, audio2, crossfade_len=140, fade_shape='sigmoid', normalization=True):
    """
    Create a smooth transition between two audio segments using advanced crossfading.
    
    Args:
        audio1: np.ndarray - First audio segment
        audio2: np.ndarray - Second audio segment
        crossfade_len: int - Length of crossfade in samples (default: 2400, or 100ms at 24kHz)
        fade_shape: str - Shape of the crossfade ('sigmoid', 'hann', 'quadratic', 'logarithmic')
        normalization: bool - Whether to normalize the crossfade region to prevent clipping
        
    Returns:
        np.ndarray: Combined audio with smooth transition
    """
    # Ensure crossfade length isn't longer than either audio segment
    crossfade_len = min(crossfade_len, len(audio1), len(audio2))
    
    # Create crossfade windows based on the selected shape
    if fade_shape == 'sigmoid':
        # Sigmoid curve (S-curve) for smoother transitions
        x = np.linspace(-6, 6, crossfade_len)
        fade_out = 1 / (1 + np.exp(x))
        fade_in = 1 - fade_out
    elif fade_shape == 'hann':
        # Hann window for more natural sounding transitions
        hann_window = signal.windows.hann(crossfade_len * 2)
        fade_out = hann_window[:crossfade_len]
        fade_in = hann_window[crossfade_len:]
    elif fade_shape == 'quadratic':
        # Quadratic curve for smoother start/end
        x = np.linspace(0, 1, crossfade_len)
        fade_out = 1 - x**2
        fade_in = x**2
    elif fade_shape == 'logarithmic':
        # Logarithmic curve for more perceptually uniform transition
        x = np.linspace(0.001, 1, crossfade_len)
        fade_out = 1 - np.log(x) / np.log(0.001)
        fade_in = np.log(x) / np.log(0.001) + 1
    else:
        # Default to linear if unrecognized shape
        fade_out = np.linspace(1.0, 0.0, crossfade_len)
        fade_in = np.linspace(0.0, 1.0, crossfade_len)
    
    # Apply crossfade
    audio1_end = audio1[-crossfade_len:]
    audio2_start = audio2[:crossfade_len]
    
    # Calculate the crossfade
    crossfade = audio1_end * fade_out + audio2_start * fade_in
    
    # Normalize the crossfade region to prevent potential clipping
    if normalization and np.max(np.abs(crossfade)) > 1.0:
        crossfade = crossfade / np.max(np.abs(crossfade))
    
    # Combine everything
    result = np.concatenate([
        audio1[:-crossfade_len],
        crossfade,
        audio2[crossfade_len:]
    ])
    
    return result
N_ZEROS = 1000
s_prev = None
reference_style = 3
for chunk in chunks:
    noise = torch.randn(1, 1, 256).to(device)
    wav, s_prev = LFinference(chunk, noise, alpha=0.4, diffusion_steps=16, embedding_scale=1 )
    wav = np.concatenate([wav , np.zeros(N_ZEROS)])
    wavs.append(wav)
    
   
# Combine all audio segments
final_audio = wavs[0]
for i in range(1, len(wavs)):
    final_audio = create_smooth_transition(final_audio, wavs[i])
#final_audio = enhance_audio(final_audio)
os.makedirs('./samples', exist_ok=True)
combined = np.concatenate(wavs, axis=0)
sf.write('./samples/combined.wav', combined, samplerate=44100, subtype='FLOAT')
