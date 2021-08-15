import torch
import sys
from tqdm import tqdm
import logging

sys.path.append('.')
from model import build_model
from dataloader import build_dataloader
from solver import build_loss, build_optimizer, build_scheduler
import pprint
from pathlib import Path
from sklearn.metrics import f1_score
import cv2
import copy
import numpy as np

class PrettyLog():
    def __init__(self, obj):
        self.obj = obj
    def __repr__(self):
        return pprint.pformat(self.obj)

class Trainer:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = cfgs.model.device
        self.train_loader, self.val_loader = build_dataloader(cfgs.dataloader)
        self.model = build_model(cfgs.model)
        self.loss_fn = build_loss(cfgs.solver)
        self.optimizer = build_optimizer(cfgs.solver, self.model)
        self.scheduler = build_scheduler(cfgs.solver, self.optimizer)
        Path(self.cfgs.output_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=f'{cfgs.output_dir}log.txt', level=logging.INFO)

        if cfgs.load_from is not None:
            print(f'load ckpt from {cfgs.load_from}')
            ckpt = torch.load(cfgs.load_from)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            print('train from scratch')

    def train(self):
        logging.info(PrettyLog(self.cfgs))
        best_acc = 0.0
        best_loss = 100.0
        best_epoch = 0
        val_acc = -1
        for epoch in range(self.cfgs.solver.max_epoch):
            train_loss, train_acc = self.__train_epoch(epoch)
            val_loss, val_acc = self.__val_epoch(epoch)
           
            if val_acc >= best_acc:
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch
                torch.save({
                    'config': self.cfgs,
                    'epoch': epoch,
                    'loss': best_loss,
                    'accuracy': best_acc,
                    'model_state_dict': self.model.state_dict(),
                    }, f'{self.cfgs.output_dir}ckpt.pth')
            
            self.scheduler.step()
            logging.info(f'epoch [{epoch}] - train_loss: {train_loss:.4f} val_loss: {val_loss:.8f}   train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}   lr: {self.scheduler._last_lr[0]:.8f}')
        logging.info(f'finish training. Best ckpt is at epoch {best_epoch}, loss: {best_loss:.8f}, acc: {best_acc:.4f}')

    def __train_epoch(self, epoch):
        self.model.train()
        running_dice_loss = 0.0
        running_bce_loss = 0.0
        runnning_acc = 0.0
        for input, target, label_128, label_64, label_32 in tqdm(self.train_loader):
            input = input.to(self.device)
            target = target.to(self.device)
            label_128 = label_128.to(self.device)
            label_64 = label_64.to(self.device)
            label_32 = label_32.to(self.device)

            self.optimizer.zero_grad()
            output, aux_128, aux_64, aux_32 = self.model(input)

            # compute loss
            soft_dice_loss, bce_loss = self.loss_fn(output, target)
            running_dice_loss += soft_dice_loss.item()
            running_bce_loss += bce_loss.item()

            # compute accuracy
            pred = output.clone()
            pred[pred >= 0.62] = 1
            pred[pred < 0.62] = 0
            runnning_acc += f1_score(pred.view(-1).detach().cpu(), target.view(-1).cpu().long())

            # auxiliary losses
            soft_dice_loss_128, bce_loss_128 = self.loss_fn(aux_128, label_128)
            soft_dice_loss_64, bce_loss_64 = self.loss_fn(aux_64, label_64)
            soft_dice_loss_32, bce_loss_32 = self.loss_fn(aux_32, label_32)
 
            loss = 0.5*(soft_dice_loss + bce_loss) \
                    + 0.3*(soft_dice_loss_128 + bce_loss_128) \
                    + 0.2*(soft_dice_loss_64 + bce_loss_64) \
                    + 0.1*(soft_dice_loss_32 + bce_loss_32) 
                    
            loss.backward()
            self.optimizer.step()

        epoch_dice_loss = running_dice_loss / len(self.train_loader)
        epoch_bce_loss = running_bce_loss / len(self.train_loader)
        epoch_acc = runnning_acc / len(self.train_loader)

        curr_lr = self.scheduler.optimizer.param_groups[0]['lr']
        # print(f'epoch {epoch} - train || dice_loss: {epoch_dice_loss:.4f} bce_loss: {epoch_bce_loss:.4f} acc: {epoch_acc:.4f} lr: {self.scheduler._last_lr[0]:.8f}')
        print(f'epoch {epoch} - train || dice_loss: {epoch_dice_loss:.4f} bce_loss: {epoch_bce_loss:.4f} acc: {epoch_acc:.4f} lr: {curr_lr:.8f}')
        

        return epoch_dice_loss + epoch_bce_loss, epoch_acc

    def __val_epoch(self, epoch):
        self.model.eval()
        running_dice_loss = 0.0
        running_bce_loss = 0.0
        runnning_acc = 0.0
        for input, target, _, _, _ in tqdm(self.val_loader):
            input = input.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output, _, _, _ = self.model(input)

            # compute loss
            soft_dice_loss, bce_loss = self.loss_fn(output, target)
            running_dice_loss += soft_dice_loss.item()
            running_bce_loss += bce_loss.item()

            # compute accuracy
            pred = output.clone()
            pred[pred >= 0.62] = 1
            pred[pred < 0.62] = 0
            runnning_acc += f1_score(pred.view(-1).detach().cpu(), target.view(-1).cpu())

        epoch_dice_loss = running_dice_loss / len(self.val_loader)
        epoch_bce_loss = running_bce_loss / len(self.val_loader)
        epoch_acc = runnning_acc / len(self.val_loader)

        print(f'epoch {epoch} - val   || dice_loss: {epoch_dice_loss:.4f} bce_loss: {epoch_bce_loss:.4f} acc: {epoch_acc:.4f}')
        return epoch_dice_loss + epoch_bce_loss, epoch_acc
