import torch
import sys
from tqdm import tqdm

sys.path.append('.')
from model import build_model
from pathlib import Path
from glob import glob
import cv2
import pickle5 as pickle
import numpy as np
from sklearn.metrics import f1_score

class Tester:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = cfgs.model.device
        self.model = build_model(cfgs.model)
        Path(f'{self.cfgs.output_dir}/submission').mkdir(parents=True, exist_ok=True)
        Path(f'{self.cfgs.output_dir}/evaluation').mkdir(parents=True, exist_ok=True)

        print(f'load ckpt from {cfgs.output_dir}')
        ckpt = torch.load(f'{cfgs.output_dir}/ckpt.pth')
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()


    def find_threshold(self):
        ann_file = open(self.cfgs.dataloader.dataset.ann_file, "rb")
        ann = pickle.load(ann_file)
        preds = list()
        targets = list()
        val_images = list()
        
        print('testing on validation set ... ')
        for val_image in ann['val']:
            image_ori = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train_shape/{val_image}')[:,:,0] / 255.
            target = cv2.imread(f'{self.cfgs.dataloader.dataset.data_folder}/img_train2/{val_image}')[:,:,0] / 255.

            image_flip_0 = cv2.flip(image_ori, 0)
            image_flip_1 = cv2.flip(image_ori, 1)
            image_flip__1 = cv2.flip(image_ori, -1)
            image = np.stack([image_ori, image_flip_0, image_flip_1, image_flip__1])

            image = torch.tensor(image).unsqueeze(1).to(self.cfgs.model.device)
            with torch.no_grad():
                pred, _, _, _ = self.model(image)
                pred = torch.sigmoid(pred)
                pred_ori, pred_flip_0, pred_flip_1, pred_flip__1 = pred

            pred_ori = pred_ori.cpu().numpy()
            pred_flip_0 = cv2.flip(pred_flip_0.cpu().numpy(), 0)
            pred_flip_1 = cv2.flip(pred_flip_1.cpu().numpy(), 1)
            pred_flip__1 = cv2.flip(pred_flip__1.cpu().numpy(), -1)
            pred = np.mean([pred_ori, pred_flip_0, pred_flip_1, pred_flip__1], axis=0)

            preds.append(pred)
            targets.append(target)
            val_images.append(val_image)
        
        preds = np.stack(preds)
        targets = np.stack(targets)

        print('finding threshold ... ')
        f1s = list()
        thresholds = np.stack(list(range(40,80)))/100
        for threshold in tqdm(thresholds):
            preds_ = preds.copy()
            preds_[preds_ >= threshold] = 1
            preds_[preds_ < threshold] = 0
            f1s.append(f1_score(preds_.reshape(-1), targets.reshape(-1)))
        f1s = np.stack(f1s)
        threshold = thresholds[f1s.argmax()]
        print(f'best f1 score is {f1s.max()} at threshold = {threshold}')

        # write valid results
        for val_image, pred, target in zip(val_images, preds, targets):
            pred[pred >= threshold] = 1
            pred[pred < threshold] = 0

            cat = np.concatenate([pred, target], axis=1)*255
            cv2.imwrite(f'{self.cfgs.output_dir}/evaluation/{val_image}', cat)

        return threshold


    def infer_tta_6(self):
        threshold = self.find_threshold()

        print(f'inferencing with threshold = {threshold}')
        for image_path in (glob(f'{self.cfgs.dataloader.dataset.data_folder}/img_test_shape/*.png')):
            image_name = image_path.split('/')[-1]

            image_ori = cv2.imread(image_path)
            image_ori = (image_ori[:,:,0]/255.)

            image_flip_0 = cv2.flip(image_ori, 0)
            image_flip_1 = cv2.flip(image_ori, 1)
            image_flip__1 = cv2.flip(image_ori, -1)
            image_rotate_90cc = cv2.rotate(image_ori, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_rotate_90c = cv2.rotate(image_ori, cv2.ROTATE_90_CLOCKWISE)
            image_rotate_180 = cv2.rotate(image_ori, cv2.ROTATE_180)
            image = np.stack([image_ori, image_flip_0, image_flip_1, image_flip__1, image_rotate_90cc, image_rotate_90c, image_rotate_180])

            image = torch.tensor(image).unsqueeze(1).to(self.cfgs.model.device)
            with torch.no_grad():
                pred, _, _, _ = self.model(image)
                pred = torch.sigmoid(pred)
                pred_ori, pred_flip_0, pred_flip_1, pred_flip__1, pred_rotate_90cc, pred_rotate_90c, pred_rotate_180 = pred

            pred_ori = pred_ori.cpu().numpy()
            pred_flip_0 = cv2.flip(pred_flip_0.cpu().numpy(), 0)
            pred_flip_1 = cv2.flip(pred_flip_1.cpu().numpy(), 1)
            pred_flip__1 = cv2.flip(pred_flip__1.cpu().numpy(), -1)
            pred_rotate_90cc = cv2.rotate(pred_rotate_90cc.cpu().numpy(), cv2.ROTATE_90_CLOCKWISE)
            pred_rotate_90c = cv2.rotate(pred_rotate_90c.cpu().numpy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
            pred_rotate_180 = cv2.rotate(pred_rotate_180.cpu().numpy(), cv2.ROTATE_180)

            pred = np.mean([pred_ori, pred_flip_0, pred_flip_1, pred_flip__1, pred_rotate_90cc, pred_rotate_90c, pred_rotate_180], axis=0)
            pred = np.stack([pred, pred, pred], axis=2)

            pred[pred >= threshold] = 255
            pred[pred < threshold] = 0
            cv2.imwrite(f'{self.cfgs.output_dir}/submission/{image_name}', pred)

        print('done')
