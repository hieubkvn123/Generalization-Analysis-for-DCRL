import torch
import pprint

# Utility functions
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def save_json_dict(json_dict, output):
    with open(output, 'w') as f:
        pprint_json_string = pprint.pformat(json_dict, compact=True).replace("'", '"')
        f.write(pprint_json_string)
        f.close()

def apply_model_to_batch(model, batch, device=None):
    # Unpack the batch
    x1, x2, x3 = batch[0].to(device), batch[1].to(device), [x.to(device) for x in batch[2]]
    
    # Apply model to batch
    y1, y2, y3 = model(x1), model(x2), [model(x) for x in x3]
    return y1, y2, y3 