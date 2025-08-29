import torch, copy
from collections import OrderedDict

path = "again_wavlm/epoch_1st_00087.pth"
out_path = "again_wavlm/epoch_1st_00087_copty_safe.pth"
overwrite = True

data = torch.load(path, map_location="cpu")
container = data["net"] if "net" in data else data

src = "style_encoder"
dst = "predictor_encoder"

if src not in container:
    raise KeyError(src + " not in checkpoint")

def clone_obj(o):
    if isinstance(o, torch.nn.Parameter):
        return torch.nn.Parameter(o.data.clone().detach(),
                                  requires_grad=o.requires_grad)
    if isinstance(o, torch.Tensor):
        return o.clone().detach()
    if isinstance(o, dict):
        return type(o)((k, clone_obj(v)) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return type(o)(clone_obj(v) for v in o)
    try:
        return copy.deepcopy(o)
    except Exception:
        return o

# remove old dst if overwrite
if dst in container:
    if overwrite:
        del container[dst]
    else:
        raise KeyError(dst + " already exists")

items = []
for k, v in container.items():
    if k == src:
        items.append((dst, clone_obj(v)))
        items.append((k, v))
    else:
        items.append((k, v))

new_container = OrderedDict(items)

if "net" in data:
    data["net"] = new_container
else:
    # jeśli data było dictem i faktycznie było "container", nadpisujemy ostrożnie:
    if isinstance(data, dict) and set(data.keys()) == set(container.keys()):
        data.clear()
        data.update(new_container)
    else:
        # zachowaj inne top-level klucze, tylko aktualizuj fragment container
        for k in list(container.keys()):
            if k in data and k not in new_container:
                del data[k]
        data.update(new_container)

torch.save(data, out_path)
print("Saved to", out_path)