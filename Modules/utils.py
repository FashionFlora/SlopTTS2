import numpy as np
import torch
import matplotlib.pyplot as plt
from munch import Munch
import os
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

nv_init = False
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def print_gpu_vram(tag):
    if False:
        global nv_init
        if not nv_init:
            nvmlInit()
            nv_init = True
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"{tag} - GPU memory occupied: {info.used//1024**2} MB.")


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(path):
    result = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            result = f.readlines()
    return result


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def length_to_mask(lengths) -> torch.Tensor:
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    # x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    x = (torch.exp(x * std + mean) ** 0.33).sum(dim=dim)
    return x


def plot_spectrogram_to_figure(
    spectrogram,
    title="Spectrogram",
    figsize=(12, 5),  # Increased width for better time resolution view
    dpi=150,  # Increased DPI for higher resolution image
    interpolation="bilinear",  # Smoother interpolation
    cmap="viridis",  # Default colormap, can change to 'magma', 'inferno', etc.
):
    """Converts a spectrogram tensor/numpy array to a matplotlib figure with improved quality."""
    plt.switch_backend("agg")  # Use non-interactive backend

    # Ensure input is a numpy array on CPU
    if isinstance(spectrogram, torch.Tensor):
        spectrogram_np = spectrogram.detach().cpu().numpy()
    elif isinstance(spectrogram, np.ndarray):
        spectrogram_np = spectrogram
    else:
        raise TypeError("Input spectrogram must be a torch.Tensor or numpy.ndarray")

    # Handle potential extra dimensions (e.g., channel dim)
    if spectrogram_np.ndim > 2:
        if spectrogram_np.shape[0] == 1:  # Remove channel dim if it's 1
            spectrogram_np = spectrogram_np.squeeze(0)
        else:
            # If multiple channels, you might want to plot only the first
            # or handle it differently (e.g., separate plots)
            spectrogram_np = spectrogram_np[0, :, :]  # Plot only the first channel
            # Or raise an error/warning:
            # raise ValueError(f"Spectrogram has unexpected shape: {spectrogram_np.shape}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  # Apply figsize and dpi

    # Ensure valid interpolation string
    valid_interpolations = [
        None,
        "none",
        "nearest",
        "bilinear",
        "bicubic",
        "spline16",
        "spline36",
        "hanning",
        "hamming",
        "hermite",
        "kaiser",
        "quadric",
        "catrom",
        "gaussian",
        "bessel",
        "mitchell",
        "sinc",
        "lanczos",
        "blackman",
    ]
    if interpolation not in valid_interpolations:
        print(f"Warning: Invalid interpolation '{interpolation}'. Using 'bilinear'.")
        interpolation = "bilinear"

    im = ax.imshow(
        spectrogram_np,
        aspect="auto",
        origin="lower",
        interpolation=interpolation,
        cmap=cmap,
    )  # Apply interpolation and cmap

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Mel Channels")  # More specific label
    plt.title(title)
    plt.tight_layout()
    # plt.close(fig) # Don't close here if returning the figure object
    return fig  # Return the figure object directly


def plot_mel_signed_difference_to_figure(
    mel_gt_normalized_np,  # Ground truth (already normalized log mel)
    mel_pred_log_np,  # Predicted (raw log mel)
    mean: float,  # Dataset mean used for normalization
    std: float,  # Dataset std used for normalization
    title="Signed Mel Log Difference (GT - Pred)",  # Updated title
    figsize=(12, 5),
    dpi=150,
    cmap="vanimo",
    max_abs_diff_clip=None,  # Optional: Clip the color range e.g., 3.0
    static_max_abs=None,  # Optional: Static max abs value for consistent color range
):
    """Plots the signed difference between two mel spectrograms using a diverging colormap."""
    plt.switch_backend("agg")

    # Ensure shapes match by trimming to the minimum length
    min_len = min(mel_gt_normalized_np.shape[1], mel_pred_log_np.shape[1])
    mel_gt_trimmed = mel_gt_normalized_np[:, :min_len]
    mel_pred_log_trimmed = mel_pred_log_np[:, :min_len]

    # Normalize the predicted log mel
    mel_pred_normalized_np = (mel_pred_log_trimmed - mean) / std

    # Calculate SIGNED difference in the *normalized* log domain
    diff = mel_gt_trimmed - mel_pred_normalized_np

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if static_max_abs is not None:
        # Use static max abs value for color limits
        vmin = -static_max_abs
        vmax = static_max_abs
    else:
        # Determine symmetric color limits centered at 0
        max_abs_val = np.max(np.abs(diff)) + 1e-9  # Add epsilon for stability
        if max_abs_diff_clip is not None:
            max_abs_val = min(
                max_abs_val, max_abs_diff_clip
            )  # Apply clipping if specified

        vmin = -max_abs_val
        vmax = max_abs_val

    im = ax.imshow(
        diff,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )  # Use 'none' for raw diff

    plt.colorbar(
        im, ax=ax, label="Signed Normalized Log Difference (GT - Pred)"
    )  # Updated label
    plt.xlabel("Frames")
    plt.ylabel("Mel Channels")
    plt.title(title)
    plt.tight_layout()
    # plt.close(fig) # Don't close if returning fig
    return fig


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(arrs)
    plt.colorbar(im, ax=ax)
    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def get_git_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError as e:
        print("Error obtaining git commit hash:", e)
        return "unknown"


def get_git_diff():
    try:
        # Run the git diff command
        diff_output = subprocess.check_output(["git", "diff"]).decode("utf-8")
        return diff_output
    except subprocess.CalledProcessError as e:
        print("Error obtaining git diff:", e)
        return ""


def save_git_diff(out_dir):
    hash = get_git_commit_hash()
    diff = get_git_diff()
    diff_file = os.path.join(out_dir, "git_state.txt")
    with open(diff_file, "w") as f:
        f.write(f"Git commit hash: {hash}\n\n")
        f.write(diff)
    print(f"Git diff saved to {diff_file}")


def duration_to_alignment(duration: torch.Tensor) -> torch.Tensor:
    """Convert a sequence of duration values to an attention matrix.

    duration -- [t]ext length
    result -- [t]ext length x [a]udio length"""
    indices = torch.repeat_interleave(
        torch.arange(duration.shape[0], device=duration.device), duration.to(torch.int)
    )
    result = torch.zeros((duration.shape[0], indices.shape[0]), device=duration.device)
    result[indices, torch.arange(indices.shape[0])] = 1
    return result


def clamped_exp(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-35, 35)
    return torch.exp(x)


def leaky_clamp(
    x_in: torch.Tensor, min_f: float, max_f: float, slope: float = 0.001
) -> torch.Tensor:
    x = x_in
    min_t = torch.full_like(x, min_f, device=x.device)
    max_t = torch.full_like(x, max_f, device=x.device)
    x = torch.maximum(x, min_t + slope * (x - min_t))
    x = torch.minimum(x, max_t + slope * (x - max_t))
    return x


class DecoderPrediction:
    def __init__(
        self,
        audio=None,
        x=None,
        y=None,
        magnitude=None,
    ):
        self.audio = audio
        self.x = x
        self.y = y
        self.magnitude = magnitude