from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import os
import numpy as np 
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from PIL import Image

from torchvision import transforms
from torchvision.transforms.transforms import ColorJitter, GaussianBlur
from torch import rand
import matplotlib.pyplot as plt

class Data_loader():
    
    def __init__(self, annotation_path, image_path, val_size, batch_size, train_transform, val_transform, shuffle = True, seed = 42, p=0.3):
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.val_size = val_size
        self.batch_size = batch_size 
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.shuffle = shuffle
        self.seed = seed
        self.p = p 

    def get_loaders(self):
        data_set = ImageDataset(self.annotation_path, self.image_path, self.train_transform, self.val_transform, self.p)
        data_size = len(data_set)
        indices = list(range(data_size))
        split = int(np.floor(self.val_size * data_size))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        self.train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(self.train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # data_set.train_idx = [6]
        # im, ms = data_set.__getitem__(6)
        #print(type(ms))
        #transforms.ToPILImage()(im.squeeze()).show()
        #plt.imsave('mask.png', ms, cmap='gray')
        
        train_loader = DataLoader(data_set, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(data_set, batch_size=self.batch_size, sampler=val_sampler)

        return train_loader, val_loader, data_set
    

class ImageDataset(Dataset):
    def __init__(self, annotaions_dir, images_dir, train_transform=None, val_transform=None, p=0.3):
        self.annotaions_dir = annotaions_dir
        self.images_dir = images_dir
        self.train_transform = train_transform
        self.coco = COCO(annotaions_dir)
        self.category_ids = self.coco.loadCats(self.coco.getCatIds())
        self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())
        self.train_idx = None
        self.val_transform = val_transform
        self.p = p
        
    def __len__(self):
        return len(self.image_ids)
        
    def  __getitem__(self, idx):
        idx_ = self.image_ids[idx]
        img = self.coco.loadImgs(idx_)[0]
        
        image_path = os.path.join(self.images_dir, img["file_name"])
        I = Image.open(image_path)
        #I = I.transpose(-1, 0, 1) #w * h * c to c * w * h
        
        annotation_ids = self.coco.getAnnIds(imgIds = img['id'])
        annotations = self.coco.loadAnns(annotation_ids)
        
        mask = np.zeros((img['height'], img['width']))
        for annotation in annotations:
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            m = m.reshape((img['height'], img['width']))
            mask = mask + m
        
        mask[mask > 0] = 1
        
        if self.train_idx is None:
            raise TypeError("train idx should be defined")
            

        did_transform = False 
        if idx in self.train_idx:
            a = rand(1).item()
            if a < self.p:
                I = self.train_transform(I)
                did_transform = True 

        if (idx not in self.train_idx) or (did_transform == False): 
            I = self.val_transform(I)        

        return I, mask

    def _init_fn(worker_id):
        np.random.seed(42 + worker_id)


if __name__ == "__main__":

    TRAIN_IMAGES_DIRECTORY = "small_data" # data/train/images"
    TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json" #roughly 280000 images
    TRAIN_ANNOTATIONS_SMALL_PATH = "data/train/annotation-small.json" #8366 images

    BATCH_SIZE = 1
    EPOCHS = 3
    LEARNING_RATE = 1e-4 
    OUTPUT_PATH = "results/"


    train_transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=7, sigma=(0.5,2)),
        #transforms.ColorJitter(0.5,0.5,0.5,0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1.0,1.0,1.0])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1.0,1.0,1.0])
    ])

    
    dataset = Data_loader(TRAIN_ANNOTATIONS_SMALL_PATH, TRAIN_IMAGES_DIRECTORY, 0.2, BATCH_SIZE, train_transform, val_transform, shuffle=True, seed=42, p=1)
    train_loader, val_loader, data_set = dataset.get_loaders()
    train_idx = dataset.train_indices

    data_set.train_idx = train_idx

    #im, ms = next(iter(train_loader))

    #transforms.ToPILImage()(im.squeeze()).show()
    #transforms.ToPILImage()(ms.squeeze()).show()
