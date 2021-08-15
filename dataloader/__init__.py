from .transforms import build_transforms
from .pixel_dataset import build_dataset
from torch.utils.data import DataLoader, Dataset


class SkeletonDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label =  self.dataset[index]
        if self.transform is not None:
            image, label, label_128, label_64, label_32 = self.transform((image, label))
            
        return image, label, label_128, label_64, label_32

    def __len__(self):
        return len(self.dataset)

def build_dataloader(cfg):
    train_ds, val_ds = build_dataset(cfg.dataset)

    # create dataloader
    train_tranforms = build_transforms(is_train=True, cfg=cfg.dataset)
    val_tranforms = build_transforms(is_train=False)

    train_loader = DataLoader(dataset=SkeletonDataset(train_ds, train_tranforms),
                    batch_size=cfg.batch_size,
                    drop_last=True,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    shuffle=True,
                    )
    val_loader = DataLoader(dataset=SkeletonDataset(val_ds, val_tranforms),
                    batch_size=cfg.batch_size,
                    drop_last=True,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    shuffle=False,
                    )

    # for sample, label in train_loader:
    #     print()

    return train_loader, val_loader

