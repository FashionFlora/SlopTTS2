import wave
import os
import numpy as np


def filter_audio_list(
    input_path="train_list.txt",
    output_path="train_list_filtered.txt",
    min_duration_sec=1.25,
    max_amplitude_threshold=0.94,
):
    """
    Filters a list of audio files based on duration and max amplitude.

    Args:
        input_path (str): Path to the input file (e.g., 'train_list.txt').
        output_path (str): Path to save the filtered list.
        min_duration_sec (float): The minimum audio duration in seconds to keep.
        max_amplitude_threshold (float): The maximum absolute amplitude
                                         (0.0 to 1.0) to keep. Files
                                         exceeding this are removed.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    kept_lines = []
    removed_count = 0
    total_count = 0

    print(f"Starting to process '{input_path}'...")
    print(f" - Minimum duration: {min_duration_sec}s")
    print(f" - Maximum amplitude: {max_amplitude_threshold}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        total_count += 1
        line = line.strip()
        if not line:
            continue

        try:
            audio_path = line.split("|")[0]

            if not os.path.exists(audio_path):
                print(f"  [Warning] File not found, skipping: {audio_path}")
                removed_count += 1
                continue

            with wave.open(audio_path, "rb") as wav_file:
                num_frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                sampwidth = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()

                # 1. Duration Check
                duration = num_frames / float(framerate)
                if duration < min_duration_sec:
                    print(
                        f"  [Removing] Short duration: {audio_path} ({duration:.2f}s)"
                    )
                    removed_count += 1
                    continue

                # 2. Amplitude Check (for 16-bit PCM, most common format)
                if sampwidth == 2:  # 16-bit audio
                    frames = wav_file.readframes(num_frames)
                    samples = np.frombuffer(frames, dtype=np.int16)

                    # Normalize to [-1.0, 1.0]
                    # 32768.0 is 2^15, the max value for a 16-bit signed integer
                    normalized_samples = samples.astype(np.float32) / 32768.0
                    max_abs_amplitude = np.max(np.abs(normalized_samples))

                    if max_abs_amplitude > max_amplitude_threshold:
                        print(
                            f"  [Removing] High amplitude: {audio_path} (Peak: {max_abs_amplitude:.2f})"
                        )
                        removed_count += 1
                        continue
                else:
                    print(
                        f"  [Info] Skipping amplitude check for {audio_path} (unsupported sample width: {sampwidth})"
                    )

                # If all checks pass, keep the line
                kept_lines.append(line + "\n")

        except wave.Error as e:
            print(
                f"  [Warning] Could not read WAV file, skipping: {audio_path} ({e})"
            )
            removed_count += 1
        except Exception as e:
            print(f"  [Error] An unexpected error occurred on line: {line}")
            print(f"          {e}")
            removed_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    print("\n--- Processing Complete ---")
    print(f"Total entries processed: {total_count}")
    print(f"Entries kept: {len(kept_lines)}")
    print(f"Entries removed (due to duration, amplitude, or errors): {removed_count}")
    print(f"Filtered list saved to: '{output_path}'")


if __name__ == "__main__":
    # Before running, ensure you have numpy installed:
    # pip install numpy

    # To run this script, make sure 'train_list.txt' is in the same
    # directory, and the paths inside it are correct.
    filter_audio_list(
        input_path="combined_final.txt",
        output_path="combined_final_filtered.txt",
        min_duration_sec=1.25,
        max_amplitude_threshold=0.93,
    )