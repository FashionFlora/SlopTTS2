import torch

checkpoint_path = "log_dir/epoch_1st_00142.pth"  # Replace with your actual checkpoint path
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Adjust depending on how the checkpoint is structured
state_dict = checkpoint.get('model', checkpoint)
state_dict = state_dict['net']
print("Filtered keys (excluding ProsodyPredictor):")
for key in state_dict:
    print(key)