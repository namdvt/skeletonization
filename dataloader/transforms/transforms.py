import random
import numpy as np
import cv2
from torchvision.transforms import Compose
import albumentations as A
import pickle
import imutils

class Normalize(object):
    def __call__(self, sample):
        image, label = sample
        image = image / 255.
        image = np.expand_dims(image, axis=0)

        # kernel = np.ones((5,5), np.uint8)
        # label = cv2.dilate(label, kernel, iterations=1)

        label = label / 255.
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        label_128 = cv2.resize(label, (128, 128), interpolation=cv2.INTER_AREA)
        label_64 = cv2.resize(label, (64, 64), interpolation=cv2.INTER_AREA)
        label_32 = cv2.resize(label, (32, 32), interpolation=cv2.INTER_AREA)

        # label_128[label_128>0] = 1
        # label_64[label_64>0] = 1
        # label_32[label_32>0] = 1

        return (image, label, label_128, label_64, label_32)


class Flip(object):
    def __call__(self, sample):
        image, label = sample
        mode = random.choice([-1, 0, 1, 2])
        if mode != 2:
            image = cv2.flip(image, mode)
            label = cv2.flip(label, mode)

        # cv2.imwrite(f'temp/{random.random()}.png', (image-label))
        return (image, label)


class Rotate(object):
    def __call__(self, sample):
        image, label = sample
        r = random.choice([0, 1, 2, 3])
        if r == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if r == 2:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
        if r == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
            label = cv2.rotate(label, cv2.ROTATE_180)

        return (image, label)


class GaussianNoise(object):
    def __init__(self):
        self.gauss_noise = A.GaussNoise(p=0.25)

    def __call__(self, sample):
        image, label = sample
        image = self.gauss_noise(image=image)['image']

        return (image, label)


class Mosaic(object):
    def __init__(self, cfg, p=0.2):
        self.p = p
        ann_file = open(cfg.ann_file, "rb")
        ann = pickle.load(ann_file)
        self.indexes = ann['train']
        self.data_folder = cfg.data_folder

    def __call__(self, sample):
        img1, label1 = sample

        if random.random() <= self.p:
            # load images
            image_names = random.choices(self.indexes, k=3)
            img2 = cv2.imread(f'{self.data_folder}/img_train_shape/{image_names[0]}')[:,:,0]
            img3 = cv2.imread(f'{self.data_folder}/img_train_shape/{image_names[1]}')[:,:,0]
            img4 = cv2.imread(f'{self.data_folder}/img_train_shape/{image_names[2]}')[:,:,0]

            label2 = cv2.imread(f'{self.data_folder}/img_train2/{image_names[0]}')[:,:,0]
            label3 = cv2.imread(f'{self.data_folder}/img_train2/{image_names[1]}')[:,:,0]
            label4 = cv2.imread(f'{self.data_folder}/img_train2/{image_names[2]}')[:,:,0]

            # concat images
            img12 = cv2.hconcat([img1, img2])
            img34 = cv2.hconcat([img3, img4])
            img1234 = cv2.vconcat([img12, img34])
            
            label12 = cv2.hconcat([label1, label2])
            label34 = cv2.hconcat([label3, label4])
            label1234 = cv2.vconcat([label12, label34])

            # resize
            img1234 = cv2.resize(img1234, dsize=(256, 256))
            label1234 = cv2.resize(label1234, dsize=(256, 256))

            return (img1234, label1234)
        return (img1, label1)


class Shift(object):
    # def __init__(self):
    #     super().__init__()
    def __call__(self, sample):
        image, label = sample
        extLeft, extRight, extTop, extBot = self.get_extreme_points(image)
        if random.choice([0, 1]):
            image, label = self.shift_left(image, label, extLeft)
        if random.choice([0, 1]):
            image, label = self.shift_right(image, label, extRight)
        if random.choice([0, 1]):
            image, label = self.shift_top(image, label, extTop)
        # if random.choice([0, 1]):
        #     image, label = self.shift_bot(image, label, extBot)

        return (image, label)

    def get_extreme_points(self, image):
        gray = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])[0]
        extRight = tuple(c[c[:, :, 0].argmax()][0])[0]
        extTop = tuple(c[c[:, :, 1].argmin()][0])[1]
        extBot = tuple(c[c[:, :, 1].argmax()][0])[1]

        return extLeft, extRight, extTop, extBot

    def shift_left(self, image, label, extLeft):
        shift_left = random.randint(0, extLeft)

        image_clipped = image[:, shift_left:]
        label_clipped = label[:, shift_left:]
        pad = np.zeros((256, shift_left))

        image_shifted = np.concatenate([image_clipped, pad], axis=1)
        label_shifted = np.concatenate([label_clipped, pad], axis=1)

        return image_shifted, label_shifted

    def shift_right(self, image, label, extRight):
        shift_right = random.randint(extRight, 255)

        image_clipped = image[:, :shift_right]
        label_clipped = label[:, :shift_right]
        pad = np.zeros((256, 256 - shift_right))

        image_shifted = np.concatenate([pad, image_clipped], axis=1)
        label_shifted = np.concatenate([pad, label_clipped], axis=1)

        return image_shifted, label_shifted

    def shift_top(self, image, label, extTop):
        shift_top = random.randint(0, extTop)

        image_clipped = image[shift_top:, :]
        label_clipped = label[shift_top:, :]
        padd = np.zeros((shift_top, 256))

        image_shifted = np.concatenate([padd, image_clipped], axis=0)
        label_shifted = np.concatenate([padd, label_clipped], axis=0)

        return image_shifted, label_shifted

    def shift_bot(self, image, label, extBot):
        shift_bot = random.randint(extBot, 256)
        padd = np.zeros((256-shift_bot,256))

        img_clipped = image[:shift_bot,:]
        img_shifted = np.concatenate([padd,img_clipped], axis=0)

        label_clipped = label[:shift_bot,:]
        label_shifted = np.concatenate([padd,label_clipped], axis=0)
        
        return img_shifted, label_shifted


class Shift2(object):
    def __call__(self, sample):
        image, label = sample
        extLeft, extRight, extTop, extBot = self.get_extreme_points(image)

        # crop
        image_cropped = image[extTop:extBot, extLeft:extRight]
        label_cropped = label[extTop:extBot, extLeft:extRight]

        # shift
        high, width = image_cropped.shape
        x = random.randint(0, 256 - width)
        y = random.randint(0, 256 - high)

        image_padded = np.zeros((256, 256))
        label_padded = np.zeros((256, 256))
        image_padded[y:y+high, x:x+width] = image_cropped
        label_padded[y:y+high, x:x+width] = label_cropped

        return (image_padded, label_padded)

    def get_extreme_points(self, image):
        gray = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # extLeft = tuple(c[c[:, :, 0].argmin()][0])[0]
        # extRight = tuple(c[c[:, :, 0].argmax()][0])[0]
        # extTop = tuple(c[c[:, :, 1].argmin()][0])[1]
        # extBot = tuple(c[c[:, :, 1].argmax()][0])[1]

        extLeft = max(0, tuple(c[c[:, :, 0].argmin()][0])[0] - 5)
        extRight = min(256, tuple(c[c[:, :, 0].argmax()][0])[0] + 5)
        extTop = max(0, tuple(c[c[:, :, 1].argmin()][0])[1] - 5)
        extBot = min(256, tuple(c[c[:, :, 1].argmax()][0])[1] + 5)

        return extLeft, extRight, extTop, extBot

def build_transforms(is_train, cfg=None):
    if is_train:
        transforms = Compose([
            # Mosaic(cfg),
            Flip(), # ok
            Rotate(), # ok
            # Flip(),
            Shift(),
            # GaussianNoise(), # ok
            Normalize()
        ])
    else:
        transforms = Compose([
            Normalize()
        ])

    return transforms
