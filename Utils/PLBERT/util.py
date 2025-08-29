import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    
    # Check if any key starts with 'module.'
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if has_module_prefix:
        print("has module prefix")
        # If 'module.' prefix exists, process the state dict
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            if name.startswith('encoder.'):
                name = name[8:]  # remove `encoder.`
            new_state_dict[name] = v
        
        # Try to delete the specific key if it exists
        try:
            del new_state_dict["embeddings.position_ids"]
        except KeyError:
            pass
        
        bert.load_state_dict(new_state_dict, strict=False)
    else:
        print("no has module prefix")
        # If no 'module.' prefix, load normally
        try:
            if "embeddings.position_ids" in state_dict:
                del state_dict["embeddings.position_ids"]
        except KeyError:
            pass
        
        bert.load_state_dict(state_dict, strict=False)
    
    return bert