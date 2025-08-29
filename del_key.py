import torch

checkpoint_path = "log_dir/epoch_1st_00142.pth"  # your checkpoint
out_path        = "log_dir/epoch_1st_00142_nopredictor.pth"

# 1. Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 2. Drill down to the net state_dict
model_dict = checkpoint.get("model", checkpoint)
net_dict   = model_dict["net"]

# 3. Identify and delete all predictor-related keys
keys_to_remove = [k for k in net_dict.keys() if "predictor" in k]
for k in keys_to_remove:
    del net_dict[k]

print(f"Removed {len(keys_to_remove)} keys:")
for k in keys_to_remove:
    print(" -", k)

# 4. Put the cleaned dict back and save
model_dict["net"]      = net_dict
checkpoint["model"]    = model_dict
torch.save(checkpoint, out_path)

print(f"Saved cleaned checkpoint to {out_path}")