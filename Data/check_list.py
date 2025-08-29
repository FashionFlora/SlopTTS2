import os
import numpy as np
from scipy.io import wavfile

def is_extreme(wav_path, threshold=95):
    """
    Return True if any sample in wav_path exceeds ±threshold% of full-scale.
    """
    try:
        _, data = wavfile.read(wav_path)
    except Exception as e:
        print(f"[SKIP]   {wav_path!r}: {e}")
        # Treat unreadable files as "extreme" so they're dropped
        return True

    samples = data.flatten().astype(np.float32)
    dt = data.dtype

    if dt == np.int16:
        max_val = 32767.0
    elif dt == np.int32:
        max_val = 2147483647.0
    elif dt == np.uint8:
        samples -= 128.0
        max_val = 127.0
    else:
        # assume float32/64 normalized to [-1,1]
        max_val = 1.0

    thresh_val = (threshold / 100.0) * max_val
    return np.any(samples >  thresh_val) or np.any(samples < -thresh_val)

def filter_to_new(input_list,
                  output_list,
                  prefix="../",
                  threshold=95):
    """
    Read input_list, drop lines whose wavs exceed ±threshold%, and
    write the remaining lines to output_list.
    """
    with open(input_list, "r", encoding="utf-8") as f:
        lines = [L for L in f if L.strip()]

    kept = []
    for L in lines:
        rel_path = L.split("|", 1)[0].strip()
        wav_path = os.path.join(prefix, rel_path)
        if not os.path.isfile(wav_path):
            print(f"[MISSING] {wav_path}")
            continue
        if is_extreme(wav_path, threshold):
            print(f"[DROP]    {wav_path}")
        else:
            kept.append(L)

    with open(output_list, "w", encoding="utf-8") as f:
        f.writelines(kept)

    print(f"\nKept {len(kept)} of {len(lines)} entries.")
    print(f"Written filtered list to: {output_list}")

if __name__ == "__main__":
    filter_to_new(
        input_list="train_list_filtered.txt",
        output_list="train_list_no_clipped_90.txt",
        prefix="../",
        threshold=90
    )