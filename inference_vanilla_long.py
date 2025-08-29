# filename: tts_long_infer.py
# Purpose: Long-form TTS inference with global BERT conditioning and smooth joins.
# What’s new vs previous versions:
# - Truly GLOBAL BERT: run PL-BERT once on the whole passage (with sliding windows
#   if >512 tokens), then slice per chunk. Diffusion and duration use the global
#   BERT slice instead of re-encoding per chunk.
# - Token-aware chunking (respects BERT 512 limit), overlap lookahead, and exact
#   tail trimming to avoid duplicated audio.
# - Boundary word dedupe (e.g., avoids repeating “Od” at the next chunk start).
# - Equal-power, correlation-aligned crossfades between chunks.
# - Smooth style evolution (EMA + noise scheduling).
# - Optional global text bias mixed into style.
# - Saves per-chunk debug tensors: output_chunk_*.txt
#
# Run this in the same environment as your project (models/utils available).

import os
import re
import time
import yaml
import math
import torch
import random
import warnings
import argparse
import numpy as np
import soundfile as sf
from munch import Munch
from collections import OrderedDict
from functools import lru_cache

import torchaudio

from models import *
from utils import *
from text_utils import TextCleaner

from Modules.diffusion.sampler import (
    DiffusionSampler,
    KarrasSchedule,
    DPMpp2MSampler,
)

warnings.simplefilter("ignore")

# -------------------------
# Repro and deterministic
# -------------------------
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
np.random.seed(0)
random.seed(0)

# -------------------------
# Globals
# -------------------------
device = "cpu"
textclenaer = TextCleaner()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
MEAN, STD = -4, 4  # kept for reference


# -------------------------
# Utility helpers
# -------------------------
def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask  # True == pad


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
            f.write(f"Type: {type(tensor)}\nValue: {tensor}\n")
    except Exception as e:
        f.write(f"Could not save variable due to exception: {repr(e)}\n")


# -------------------------
# Smart crossfade with alignment search
# -------------------------
def _smart_crossfade_pair(prev, nxt, sr, fade_ms=80, search_ms=35, match_rms=True):
    prev = prev.astype(np.float32, copy=False)
    nxt = nxt.astype(np.float32, copy=False)
    if len(prev) == 0:
        return nxt
    if len(nxt) == 0:
        return prev

    fade_len = max(16, int(sr * fade_ms / 1000))
    if len(prev) < fade_len or len(nxt) < fade_len:
        # Too short for crossfade; concat
        return np.concatenate([prev, nxt], axis=0)

    search = max(0, int(sr * search_ms / 1000))
    search = min(search, max(0, len(nxt) - fade_len))

    tail = prev[-fade_len:]
    head = nxt[: fade_len + search]

    # Correlation-based best offset
    best_o = 0
    best_c = -1e9
    denom_tail = np.sqrt((tail * tail).sum() + 1e-8)
    for o in range(0, search + 1):
        seg = head[o : o + fade_len]
        denom_seg = np.sqrt((seg * seg).sum() + 1e-8)
        c = float((tail * seg).sum()) / (denom_tail * denom_seg + 1e-8)
        if c > best_c:
            best_c = c
            best_o = o

    seg = nxt[best_o : best_o + fade_len]
    if match_rms:
        rms_tail = np.sqrt(np.mean(tail * tail) + 1e-8)
        rms_seg = np.sqrt(np.mean(seg * seg) + 1e-8)
        gain = 1.0 if rms_seg < 1e-6 else float(rms_tail / rms_seg)
        seg = seg * gain

    # Equal-power crossfade
    t = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
    w1 = np.cos(0.5 * np.pi * t).astype(np.float32)
    w2 = np.sin(0.5 * np.pi * t).astype(np.float32)
    overlap = tail * w1 + seg * w2

    out = np.concatenate(
        [prev[:-fade_len], overlap, nxt[best_o + fade_len :]], axis=0
    )
    return out


def smart_crossfade_concat(wavs, fade_ms=80, search_ms=35, sr=44100):
    if not wavs:
        return np.array([], dtype=np.float32)
    out = wavs[0].astype(np.float32)
    for w in wavs[1:]:
        out = _smart_crossfade_pair(
            out, w, sr=sr, fade_ms=fade_ms, search_ms=search_ms
        )
    return out


# -------------------------
# Phonemizer (Polish)
# -------------------------
import phonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="pl", preserve_punctuation=True, with_stress=True
)


# -------------------------
# Tokenization helpers (same pipeline as inference)
# -------------------------
@lru_cache(maxsize=8192)
def tokens_from_text(text):
    if not text:
        return []
    ps = global_phonemizer.phonemize([text])[0].split()
    ps = " ".join(ps)
    tokens = textclenaer(ps)
    return list(tokens)  # no BOS here


@lru_cache(maxsize=8192)
def count_tokens_for_text(text, include_bos=True):
    t = tokens_from_text(text)
    return len(t) + (1 if include_bos else 0)


def split_text_to_fit_tokens(text, max_tokens, include_bos=True):
    words = text.strip().split()
    if not words:
        return []
    out, cur = [], []
    for w in words:
        cand = (" ".join(cur + [w])).strip()
        if count_tokens_for_text(cand, include_bos=include_bos) <= max_tokens:
            cur.append(w)
        else:
            if cur:
                out.append(" ".join(cur))
            cur = [w]
            if count_tokens_for_text(" ".join(cur), include_bos=include_bos) > max_tokens:
                out.append(" ".join(cur))
                cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def _normalize_word(w: str) -> str:
    core = re.sub(r"[^\w]+", "", w, flags=re.UNICODE)
    return core.casefold()


def _first_word(s: str) -> str:
    for tok in s.strip().split():
        nw = _normalize_word(tok)
        if nw:
            return nw
    return ""


def _last_word(s: str) -> str:
    for tok in reversed(s.strip().split()):
        nw = _normalize_word(tok)
        if nw:
            return nw
    return ""


def _remove_leading_first_word(s: str) -> str:
    parts = s.strip().split()
    if not parts:
        return s
    return " ".join(parts[1:]).strip()


def take_prefix_with_token_budget(text, max_lookahead_tokens, full_text_budget_tokens, base):
    if max_lookahead_tokens <= 0:
        return ""
    words = text.strip().split()
    acc = []
    for w in words:
        cand = (" ".join(acc + [w])).strip()
        if count_tokens_for_text(cand, include_bos=False) > max_lookahead_tokens:
            break
        if (
            count_tokens_for_text(f"{base} {cand}".strip(), include_bos=True)
            > full_text_budget_tokens
        ):
            break
        acc.append(w)
    return " ".join(acc)


def chunk_with_overlap_token_aware(text, max_tokens, overlap_tokens):
    """
    Token-aware chunking. Produces list of dicts:
    {
      'base': base_text,
      'lookahead': lookahead_prefix_from_next,
      'full': base + ' ' + lookahead,
      'ending_punct': ['.', '?', '!', '']
    }
    Ensures tokens(full) <= max_tokens for every chunk.
    Removes duplicated boundary word (last(prev)==first(next)).
    """
    parts = re.split(r"([.!?…]+)", text)
    sents = []
    for i in range(0, len(parts), 2):
        s = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if not s:
            continue
        sents.append((s + punct).strip())

    bases = []
    cur = ""
    for s in sents:
        candidate = (f"{cur} {s}".strip()) if cur else s
        if count_tokens_for_text(candidate, include_bos=True) <= max_tokens:
            cur = candidate
        else:
            if cur:
                bases.append(cur)
                cur = ""
            if count_tokens_for_text(s, include_bos=True) <= max_tokens:
                cur = s
            else:
                parts_fit = split_text_to_fit_tokens(s, max_tokens, include_bos=True)
                for k, p in enumerate(parts_fit):
                    if k < len(parts_fit) - 1:
                        bases.append(p)
                    else:
                        cur = p
    if cur:
        bases.append(cur)

    # boundary word dedupe
    for i in range(len(bases) - 1):
        lw = _last_word(bases[i])
        fw = _first_word(bases[i + 1])
        if lw and fw and lw == fw:
            bases[i + 1] = _remove_leading_first_word(bases[i + 1])

    annotated = []
    for i, base in enumerate(bases):
        ending_punct = ""
        bt = base.rstrip()
        if bt.endswith("?"):
            ending_punct = "?"
        elif bt.endswith("!"):
            ending_punct = "!"
        elif bt.endswith("."):
            ending_punct = "."

        nxt = bases[i + 1] if i + 1 < len(bases) else ""
        lookahead = ""
        if nxt and overlap_tokens > 0:
            full_budget = max_tokens
            la = take_prefix_with_token_budget(
                nxt,
                max_lookahead_tokens=overlap_tokens,
                full_text_budget_tokens=full_budget,
                base=base,
            )
            lookahead = la
        full = (f"{base} {lookahead}".strip()) if lookahead else base
        if count_tokens_for_text(full, include_bos=True) > max_tokens:
            lookahead = ""
            full = base
        annotated.append(
            {
                "base": base,
                "lookahead": lookahead,
                "full": full,
                "ending_punct": ending_punct,
            }
        )
    return annotated


def find_subsequence(seq, sub, start_idx=0):
    """
    Find first index >= start_idx where sub occurs in seq. Returns -1 if not found.
    Both seq and sub are lists of ints.
    """
    if not sub:
        return start_idx
    n, m = len(seq), len(sub)
    i = start_idx
    while i + m <= n:
        if seq[i : i + m] == sub:
            return i
        i += 1
    return -1


def map_chunks_to_global_token_spans(annotated, tokens_global):
    """
    For each chunk in annotated, compute token start and length wrt global tokens.
    We anchor on base first, then extend to full (base + lookahead).
    Enforces monotonic forward mapping to avoid earlier duplicates.
    """
    spans = []
    cursor = 0
    for ch in annotated:
        base_toks = tokens_from_text(ch["base"])
        full_toks = tokens_from_text(ch["full"])
        start = find_subsequence(tokens_global, base_toks, start_idx=cursor)
        if start < 0:
            # fallback: try full directly
            start = find_subsequence(tokens_global, full_toks, start_idx=cursor)
            if start < 0:
                # last resort: anchor on first 6 tokens of base
                base_head = base_toks[:6]
                start = find_subsequence(tokens_global, base_head, start_idx=cursor)
                if start < 0:
                    # give up: align at cursor
                    start = cursor
                    base_toks = base_head
                    full_toks = base_head
        # full span
        tok_start = start
        tok_len = len(full_toks)
        spans.append(
            {
                "tok_start": tok_start,
                "tok_len": tok_len,
                "base_len": len(base_toks),
            }
        )
        # advance cursor to after base (keeps lookahead for next)
        cursor = tok_start + max(1, len(base_toks))
    return spans


# -------------------------
# Model loading
# -------------------------
def load_all_models(config_path, ckpt_path):
    config = yaml.safe_load(open(config_path))

    ASR_config = config.get("ASR_config", False)
    ASR_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    from Utils.PLBERT.util import load_plbert

    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    model = build_model(
        recursive_munch(config["model_params"]),
        text_aligner,
        pitch_extractor,
        plbert,
    )
    _ = [model[k].to(device) for k in model]
    _ = [model[k].eval() for k in model]

    params_whole = torch.load(ckpt_path, map_location="cpu")
    params = params_whole["net"]
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except Exception:
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    _ = [model[k].eval() for k in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=DPMpp2MSampler(s_churn=0.0, s_tmin=0.5, s_tmax=3.0, s_noise=1.0),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    return config, model, sampler


# -------------------------
# GLOBAL BERT (sliding window, overlap-add)
# -------------------------
def run_global_bert_embeddings(model, tokens_global, max_pos):
    """
    tokens_global: list[int] (no BOS), length T
    Returns bert_global: [1, T, hidden] on device
    Uses sliding windows with 25% overlap (window= max_pos, hop ~0.75*window).
    """
    T = len(tokens_global)
    if T == 0:
        hidden = int(model.bert.config.hidden_size)
        return torch.zeros(1, 0, hidden, device=device)

    window = max_pos
    hop = max(1, int(window * 0.75))  # 25% overlap
    # create accumulation buffers
    hidden = int(model.bert.config.hidden_size)
    acc = torch.zeros(T, hidden, device=device)
    wsum = torch.zeros(T, 1, device=device)

    # cosine window for overlap-add
    def cosine_win(n):
        t = torch.linspace(0, math.pi, steps=n, device=device)
        return (0.5 - 0.5 * torch.cos(t)).unsqueeze(-1)  # [n,1]

    pos = 0
    while pos < T:
        end = min(pos + window, T)
        sl = tokens_global[pos:end]
        # build tensors
        tok = torch.tensor(sl, dtype=torch.long, device=device).unsqueeze(0)  # [1,L]
        lens = torch.tensor([len(sl)], dtype=torch.long, device=device)
        mask = length_to_mask(lens).to(tok.device)
        # run BERT
        with torch.no_grad():
            bert_chunk = model.bert(tok, attention_mask=(~mask).int())  # [1,L,H]
        w = cosine_win(bert_chunk.shape[1])  # [L,1]
        # accumulate
        acc[pos:end] += bert_chunk.squeeze(0) * w
        wsum[pos:end] += w
        if end == T:
            break
        pos = pos + hop

    bert_global = (acc / wsum.clamp_min(1e-8)).unsqueeze(0)  # [1,T,H]
    return bert_global


# -------------------------
# Inference core (uses global BERT slice)
# -------------------------
def LFinference(
    chunk_text,
    chunk_tokens,  # list[int], no BOS
    bert_slice,  # [1, Tc, H]
    model,
    sampler,
    s_prev,
    noise,
    alpha=0.7,
    diffusion_steps=16,
    embedding_scale=1.2,
    attn_window=15,
    use_global_text_bias=True,
    out_txt_path="output.txt",
    trim_tail_tokens=0,
    ending_punct="",
    next_exists=False,
):
    """
    Returns:
      - waveform (np.ndarray)
      - s_pred (torch.Tensor) style vec [1, 256] for chaining
    """
    if hasattr(model, "predictor"):
        model.predictor.attn_window = int(attn_window)

    # Build tensors (NO BOS to keep same length as bert_slice)
    tokens_tensor = torch.LongTensor(chunk_tokens).to(device).unsqueeze(0)
    input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(tokens_tensor.device)
    text_mask = length_to_mask(input_lengths).to(tokens_tensor.device)

    with torch.no_grad(), open(out_txt_path, "w", encoding="utf-8") as f:
        save_tensor(f, "tokens", tokens_tensor)
        save_tensor(f, "input_lengths", input_lengths)
        save_tensor(f, "text_mask", text_mask)

        # text encoder (local, per chunk)
        t_en = model.text_encoder(tokens_tensor, input_lengths, text_mask)
        save_tensor(f, "t_en", t_en)

        # Use global BERT slice directly
        bert_dur = bert_slice  # [1, T, H]
        save_tensor(f, "bert_dur_global_slice", bert_dur)

        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        save_tensor(f, "d_en", d_en)

        # diffusion style sampling on global BERT context (slice)
        s_pred = sampler(
            noise=noise,
            embedding=bert_dur,
            num_steps=int(diffusion_steps),
            embedding_scale=float(embedding_scale),
            embedding_mask_proba=0.0,
        ).squeeze(1)  # [B, 256]
        save_tensor(f, "s_pred_raw", s_pred)

        # optional global text bias mixed into style
        if use_global_text_bias:
            valid = (~text_mask).float().unsqueeze(-1)  # [B, T, 1]
            denom = valid.sum(dim=1).clamp_min(1.0)
            g_text = (bert_dur * valid).sum(dim=1) / denom  # [B, H]
            if not hasattr(model, "global_text_proj"):
                model.global_text_proj = torch.nn.Linear(
                    g_text.size(-1), s_pred.size(-1)
                ).to(device)
                torch.nn.init.xavier_uniform_(model.global_text_proj.weight)
                torch.nn.init.zeros_(model.global_text_proj.bias)
                model.global_text_proj.eval()
            g_bias = torch.tanh(model.global_text_proj(g_text))  # [B,256]
            s_pred = 0.8 * s_pred + 0.2 * g_bias
            save_tensor(f, "s_pred_with_text_bias", s_pred)

        if s_prev is not None:
            s_pred = alpha * s_prev + (1.0 - alpha) * s_pred
            save_tensor(f, "s_pred_after_mix", s_pred)

        ref = s_pred[:, :128]
        s = s_pred[:, 128:]
        save_tensor(f, "ref", ref)
        save_tensor(f, "s", s)

        # duration encoder (Conformer) -> token-level features from d_en
        d_tok = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)  # [B,T,d]
        style_exp = s.unsqueeze(1).expand(-1, d_tok.size(1), -1)
        d = torch.cat([d_tok, style_exp], dim=-1)  # [B,T,d_hid+sty]
        x, _ = model.predictor.lstm(d)
        save_tensor(f, "x_lstm", x)

        duration_raw = model.predictor.duration_proj(
            torch.nn.functional.dropout(x, p=0.0, training=False)
        )  # [B,T,max_dur]
        duration = torch.sigmoid(duration_raw).sum(axis=-1)  # [B,T]
        pred_dur = (
            torch.round(duration.squeeze()).clamp(min=1).to(torch.long)
        )  # [T]

        # Cadence boost (smaller if another chunk follows)
        end_char = (ending_punct or "").strip()
        if next_exists:
            end_bump = 1
            if end_char == "?":
                end_bump = 3
            elif end_char == "!":
                end_bump = 3
            elif end_char == ".":
                end_bump = 2
        else:
            end_bump = 3
            if end_char == "?":
                end_bump = 8
            elif end_char == "!":
                end_bump = 6
            elif end_char == ".":
                end_bump = 4
        if pred_dur.numel() > 0:
            pred_dur[-1] = pred_dur[-1] + end_bump
        save_tensor(f, "pred_dur", pred_dur)

        # build monotonic alignment [T, F] from durations
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

        total_frames = int(pred_dur_arr.sum())
        total_frames = max(total_frames, 1)

        pred_aln_trg = torch.zeros((tokens_len, total_frames), dtype=torch.float32)
        c_frame = 0
        for i in range(tokens_len):
            dur_i = int(pred_dur_arr[i])
            if dur_i <= 0:
                dur_i = 1
            end = min(c_frame + dur_i, total_frames)
            pred_aln_trg[i, c_frame:end] = 1.0
            c_frame += dur_i
            if c_frame >= total_frames:
                break

        en = model.predictor.compute_en(
            d_tok=d,
            alignment=pred_aln_trg.unsqueeze(0).to(device),
            style=s,
            text_mask=text_mask,
        )
        save_tensor(f, "en", en)

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        save_tensor(f, "F0_pred", F0_pred)
        save_tensor(f, "N_pred", N_pred)

        t_en_aligned = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        save_tensor(f, "t_en_aligned", t_en_aligned)

        ref_in = ref
        out, _, _ = model.decoder(t_en_aligned, F0_pred, N_pred, ref_in)

        # Trim tail that corresponds to lookahead tokens to avoid duplication.
        if trim_tail_tokens > 0:
            k = int(trim_tail_tokens)
            k = min(k, len(pred_dur_arr))
            trim_frames = int(pred_dur_arr[-k:].sum())
            keep_frames = max(total_frames - trim_frames, 0)
            save_tensor(f, "trim_tail_tokens", torch.tensor([k]))
            save_tensor(f, "trim_frames", torch.tensor([trim_frames]))
            save_tensor(f, "keep_frames", torch.tensor([keep_frames]))
            if keep_frames > 0 and total_frames > 0:
                spf = float(out.shape[-1]) / float(total_frames)
                keep_samples = int(round(spf * keep_frames))
                out = out[..., :keep_samples]

    return out.squeeze().detach().cpu().numpy().astype(np.float32), s_pred


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Long TTS inference (global BERT)")
    parser.add_argument(
        "--config", type=str, default="Configs/config.yml", help="Path to YAML config"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="again_wavlm_diff/epoch_2nd_00032.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Rok 2034. Cały świat leży w gruzach. Ludzkość niemal w całości "
            "zginęła. Promieniowanie sprawia, że na wpół zburzone miasta nie "
            "są zdatne do życia. A poza ich granicami podobno zaczynają się "
            "bezkresne wypalone pustynie i gęstwiny zmutowanych lasów. Ale "
            "nikt nie wie na pewno, co tam jest. Cywilizacja ginie. Pamięć o "
            "dawnej wielkości człowieka obrasta w bajki i zmienia się w "
            "legendy."
        ),
        help="Input passage text",
    )
    parser.add_argument(
        "--out", type=str, default="./samples/combined_long_bert.wav", help="Output WAV path"
    )
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    parser.add_argument(
        "--fade_ms", type=int, default=80, help="Crossfade length in ms between chunks"
    )
    parser.add_argument(
        "--search_ms",
        type=int,
        default=35,
        help="Max alignment search window (ms) for smarter crossfade",
    )
    parser.add_argument(
        "--attn_window",
        type=int,
        default=15,
        help="Predictor cross-attention window (frames) for inference",
    )
    parser.add_argument(
        "--diff_steps", type=int, default=44, help="Diffusion steps during style sampling"
    )
    parser.add_argument(
        "--embed_scale",
        type=float,
        default=1.2,
        help="Classifier-free guidance scale for diffusion",
    )
    parser.add_argument(
        "--alpha_min", type=float, default=0.2, help="Min EMA alpha for s_prev mixing"
    )
    parser.add_argument(
        "--alpha_max", type=float, default=0.7, help="Max EMA alpha for s_prev mixing"
    )
    parser.add_argument(
        "--noise_sigma_max", type=float, default=0.9, help="Initial noise multiplier"
    )
    parser.add_argument(
        "--noise_sigma_min", type=float, default=0.4, help="Final noise multiplier"
    )
    parser.add_argument(
        "--save_dir", type=str, default=".", help="Directory to save per-chunk debug txt"
    )
    parser.add_argument(
        "--use_text_bias", action="store_true", help="Mix in global text bias to style"
    )
    parser.add_argument(
        "--bert_token_limit",
        type=int,
        default=0,
        help=(
            "Hard cap for tokens per BERT input "
            "(<= model.bert.config.max_position_embeddings). 0=auto."
        ),
    )
    parser.add_argument(
        "--overlap_tokens",
        type=int,
        default=48,
        help="How many tokens to borrow from next chunk as lookahead.",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading models...")
    _, model, sampler = load_all_models(args.config, args.ckpt)
    print("Models loaded.")

    # Discover latent channels for diffusion noise (style_dim * 2).
    style_dim2 = getattr(model.diffusion, "channels", None)
    if style_dim2 is None:
        style_dim2 = getattr(getattr(model, "diffusion", None), "unet", None)
        style_dim2 = getattr(style_dim2, "channels", 256)
    if not isinstance(style_dim2, int) or style_dim2 <= 0:
        style_dim2 = 256  # safe fallback

    # Determine BERT max tokens
    bert_max_pos = int(getattr(model.bert.config, "max_position_embeddings", 512))
    if args.bert_token_limit > 0:
        max_tokens = min(args.bert_token_limit, bert_max_pos)
    else:
        max_tokens = max(8, bert_max_pos - 2)  # small margin

    # Token-aware chunking (with boundary dedupe and lookahead)
    annotated = chunk_with_overlap_token_aware(
        args.text, max_tokens=max_tokens, overlap_tokens=max(0, args.overlap_tokens)
    )
    if not annotated:
        print("No chunks produced from text.")
        sf.write(args.out, np.array([], dtype=np.float32), args.sr)
        return

    # GLOBAL tokens and GLOBAL BERT embeddings
    tokens_global = tokens_from_text(args.text)  # no BOS
    bert_global = run_global_bert_embeddings(model, tokens_global, max_pos=bert_max_pos)
    # Map chunks to global token spans
    spans = map_chunks_to_global_token_spans(annotated, tokens_global)

    wavs = []
    s_prev = None

    t0 = time.time()
    for i, (item, span) in enumerate(zip(annotated, spans)):
        full_text = item["full"]
        ending_punct = item["ending_punct"]
        has_next = i < (len(annotated) - 1)

        # chunk tokens (no BOS to match bert_slice length)
        chunk_tokens = tokens_from_text(full_text)
        tok_start = int(span["tok_start"])
        tok_len = int(span["tok_len"])

        # build bert slice
        bert_slice = bert_global[:, tok_start : tok_start + tok_len, :]  # [1,Tc,H]

        # tokens of lookahead to trim off the tail
        # (compute using difference between full and base lengths)
        base_len = int(span["base_len"])
        lookahead_len = max(0, tok_len - base_len)
        trim_tail = lookahead_len

        # style evolution schedule
        t = i / max(1, len(annotated) - 1)
        alpha = args.alpha_min + (args.alpha_max - args.alpha_min) * t
        sigma = args.noise_sigma_max - (args.noise_sigma_max - args.noise_sigma_min) * t
        noise = torch.randn(1, 1, style_dim2, device=device) * float(sigma)

        wav, s_prev = LFinference(
            chunk_text=full_text,
            chunk_tokens=chunk_tokens,
            bert_slice=bert_slice,
            model=model,
            sampler=sampler,
            s_prev=s_prev,
            noise=noise,
            alpha=alpha,
            diffusion_steps=args.diff_steps,
            embedding_scale=args.embed_scale,
            attn_window=args.attn_window,
            use_global_text_bias=args.use_text_bias,
            out_txt_path=os.path.join(args.save_dir, f"output_chunk_{i}.txt"),
            trim_tail_tokens=trim_tail,
            ending_punct=ending_punct,
            next_exists=has_next,
        )
        wavs.append(wav)

    final_audio = smart_crossfade_concat(
        wavs, fade_ms=args.fade_ms, search_ms=args.search_ms, sr=args.sr
    )
    sf.write(args.out, final_audio, samplerate=args.sr, subtype="FLOAT")

    print(
        f"Saved {args.out} and per-chunk debug files to {args.save_dir}. "
        f"Took {time.time() - t0:.2f}s for {len(annotated)} chunks. "
        f"(Global tokens: {len(tokens_global)}, BERT max pos: {bert_max_pos})"
    )


if __name__ == "__main__":
    main()