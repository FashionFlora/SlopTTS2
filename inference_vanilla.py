# filename: tts_save_tensors.py
# Purpose: same pipeline but extended to save intermediate tensors/vars to output.txt
# Ensure you run this in the same environment where your models and utils are available.

import os
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import soundfile as sf
import random
random.seed(0)
import re
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
device = "cuda"

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
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
    "again_wavlm_diff/epoch_2nd_00020.pth", map_location="cpu"
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

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=DPMpp2MSampler(s_churn=0.00, s_tmin=0.5, s_tmax=3.0, s_noise=1.0),
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
):
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

        s_pred = sampler(
            noise,
            embedding=bert_dur_first[0].unsqueeze(0),
            num_steps=64,
            embedding_scale=1.0,
            embedding_mask_proba = 0.0
        ).squeeze(0)
        save_tensor(f, "s_pred", s_pred)

        if s_prev is not None:
            s_pred = alpha * s_prev + (1 - alpha) * s_pred
            save_tensor(f, "s_pred_after_mix", s_pred)

        ref = s_pred[:, :128]
        s = s_pred[:, 128:]

        save_tensor(f, "ref", ref)
        save_tensor(f, "s", s)

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        style_exp = s.unsqueeze(1).expand(-1, d.size(1), -1)
        d = torch.cat([d, style_exp], dim=-1)
 
        x, _ = model.predictor.lstm(d)
        save_tensor(f, "x", x)

        duration_raw = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration_raw).sum(axis=-1)

        pred_dur = torch.round(duration.squeeze()).clamp(min=1).to(torch.long)
        pred_dur[-1] += 5
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



        en = model.predictor.compute_en(
            d_tok=d,
            alignment=pred_aln_trg.unsqueeze(0).to(device),
            style=s,
            text_mask=text_mask,
        )
        #en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        print(F0_pred)
        print(N_pred)
        save_tensor(f, "F0_pred", F0_pred)
        save_tensor(f, "N_pred", N_pred)
        t_en_aligned = (t_en @ pred_aln_trg.unsqueeze(0).to(device))

        ref_in = ref.squeeze().unsqueeze(0).to(device)
        # ref_in = torch.rand_like(ref_in)
        # t_en_aligned = torch.rand_like(t_en_aligned)
        out,_,_ = model.decoder(t_en_aligned, F0_pred, N_pred, ref_in)

    # return waveform and style vector for chaining
    return out.squeeze().cpu().numpy(), s_pred

def smart_text_chunking(passage, target_length=150):
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

if __name__ == "__main__":
    passage = '''Napięcie rosło z każdą godziną. Najlepsi żołnierze, doskonale wyposażeni i specjalnie przygotowywani do takich zadań, czuwali na okrągło. Talia kart, przy których zabijało się czas między jednym a drugim alarmem, już drugą dobę pokrywała się kurzem w szufladzie biurka w wartowni. Zwyczajowe pogaduszki przeszły w ciche, niespokojne rozmowy, a te – w ciężkie milczenie: każdy miał nadzieję, że pierwszy usłyszy echo kroków powracającej karawany. Zbyt wiele od niej zależało.'''

    chunks = smart_text_chunking(passage, target_length=250)
    wavs = []
    s_prev = None
    os.makedirs("./samples", exist_ok=True)

    for i, chunk in enumerate(chunks):
        noise = torch.randn(1, 1, 256).to(device)
        wav, s_prev = LFinference(
            chunk, s_prev, noise, alpha=0.3, diffusion_steps=12, embedding_scale=1,
            out_txt_path=f"output_chunk_{i}.txt"
        )
        wavs.append(wav)

    # Simple concatenation of chunks (no smooth crossfade).
    # Each chunk already had N_ZEROS zeros appended, so concatenating
    # preserves short silences between chunks without complex fading.
    if len(wavs) == 0:
        final_audio = np.array([], dtype=np.float32)
    else:
        final_audio = np.concatenate(wavs, axis=0)

    sf.write("./samples/combined1.wav", final_audio, samplerate=44100, subtype="FLOAT")
    print("Saved ./samples/combined1.wav and output_chunk_*.txt files.")