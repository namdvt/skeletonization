from tools.train import Trainer
from tools.test import Tester
import yaml
from easydict import EasyDict as edict

if __name__=='__main__':
    with open('configs/unet_att.yaml') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
        cfgs = edict(cfgs)

    trainer = Trainer(cfgs)
    trainer.train()

    tester = Tester(cfgs)
    tester.infer_tta_6()
    print()