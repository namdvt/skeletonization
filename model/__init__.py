from operator import mod
from .unet_att import UnetAttention

def build_model(cfg):
    print(f'loading model {cfg.name}')
    if cfg.name == 'unet_att':
        model = UnetAttention()
    else:
        print("unsupport model type")
        model = None

    model = model.to(cfg.device)       
    return model

