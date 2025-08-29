import torch
import sys
import os

def modify_and_save_checkpoint(file_path, output_file_path=None):
    """
    Loads a .pth file, renames the 'model' key to 'net' if it exists,
    and saves the modified checkpoint.

    Args:
        file_path (str): The path to the input .pth file.
        output_file_path (str, optional): The path to save the modified .pth file.
                                         If None, defaults to original_filename_renamed.pth.
    """
    try:
        print(f"Attempting to load: {file_path}...")
        # Load the checkpoint to CPU to avoid GPU issues
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        print(f"Successfully loaded: {file_path}")

        if not isinstance(checkpoint, dict):
            print(
                f"The loaded data from '{file_path}' is not a dictionary. "
                "Cannot proceed with key renaming."
            )
            print(f"Type of loaded data: {type(checkpoint)}")
            return

        print("\nOriginal top-level keys:")
        if not checkpoint:
            print("(The dictionary is empty)")
        else:
            for key in checkpoint.keys():
                print(f"- {key}")

        modified = False
        if "model" in checkpoint:
            print("\nFound 'model' key. Renaming to 'net'...")
            # Store the value of 'model'
            model_state_dict = checkpoint["model"]
            # Remove the 'model' key
            del checkpoint["model"]
            # Add the 'net' key with the stored value
            checkpoint["net"] = model_state_dict
            modified = True
            print("'model' key successfully renamed to 'net'.")
        else:
            print("\nNo 'model' key found. No changes made to keys.")

        if modified:
            if output_file_path is None:
                base, ext = os.path.splitext(file_path)
                output_file_path = f"{base}_renamed{ext}"

            print(f"\nSaving modified checkpoint to: {output_file_path}...")
            torch.save(checkpoint, output_file_path)
            print(f"Successfully saved modified checkpoint.")

            print("\nTop-level keys in the new saved file:")
            if not checkpoint:
                print("(The dictionary is empty)")
            else:
                for key in checkpoint.keys():
                    print(f"- {key}")
            if "net" in checkpoint and isinstance(checkpoint["net"], dict):
                print("\nKeys in 'net' (formerly 'model'):")
                if not checkpoint["net"]:
                    print("(The 'net' dictionary is empty)")
                else:
                    for sub_key in checkpoint["net"].keys():
                        print(f"  - {sub_key}")
        else:
            print("No modifications were made, so no new file was saved.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "This could be due to a corrupted file, an incompatible PyTorch version,"
        )
        print("or the file not being a standard PyTorch save file.")


if __name__ == "__main__":
    input_path = "epoch_00070.pth"
    output_path = "epoch_00070changed.pth"
    modify_and_save_checkpoint(input_path, output_path)