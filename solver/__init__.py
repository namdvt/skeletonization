from .loss import Loss
import torch.optim as optim

def build_loss(cfg):
    if cfg.loss == 'dice':
        loss_fn = Loss()
    else:
        print("unsupport loss function")
        loss_fn = None

    return loss_fn

def build_optimizer(cfg, model):
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                                lr=cfg.base_lr,
                                momentum=cfg.momentum, 
                                nesterov=True,
                                weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg.base_lr, 
                                betas=(0.9, 0.999), 
                                eps=1e-8,
                                weight_decay=cfg.weight_decay)
    else:
        print("unsupport optimizer function")
        optimizer = None

    return optimizer

def build_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cfg.T_0, cfg.T_mul)
    elif cfg.lr_scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', cfg.factor, cfg.patience)
    elif cfg.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.milestones, cfg.gamma)
    else:
        print("unsupport scheduler function")
        scheduler = None

    return scheduler