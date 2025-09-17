import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class singleClass(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    # I will have to change this function to work with the json/ mask format
    # i have my data in
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0]=1.0 #makes binary (for sigmoid)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask
    

class fillerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, classes=["particle","dark_spot"], transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.classes = classes
        
        self.images = os.listdir(image_dir)
        
        # Get the masks for each class
        class_folder_dir = []
        class_mask_lists = []
        for seg_class in classes:
            class_mask_dir = mask_dir / seg_class
            class_folder_dir.append(class_mask_dir)

            class_mask_lists.append(os.listdir(class_mask_dir))
            
        self.class_folder_dir = class_folder_dir
        self.class_masks_lists = class_mask_lists
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        # build per-class mask list (H, W) – one per class
        img_name = self.images[index]
        masks_list = []
        for class_dir in self.class_folder_dir:
            mpath = class_dir / img_name
            if mpath.exists():
                m = np.array(Image.open(mpath).convert("L"), dtype=np.float32)
                m = (m > 0).astype(np.float32) # <- binarizes
            else:
                m = np.zeros(image.shape[:2], dtype=np.float32)
            masks_list.append(m)

        # IMPORTANT: use 'masks=' (list) for Albumentations, not 'mask='
        if self.transform is not None:
            augmented = self.transform(image=image, masks=masks_list)
            image, masks_list = augmented["image"], augmented["masks"]
        else:
            # keep as list for consistent stacking below
            pass

        # Stack AFTER transforms
        # If ToTensorV2 is used, masks_list items are torch.Tensors (H, W).
        try:
            import torch
            if isinstance(masks_list[0], torch.Tensor):
                masks = torch.stack(masks_list, dim=0)          # (C, H, W) for training
            else:
                masks = np.stack(masks_list, axis=-1)           # (H, W, C)
        except ImportError:
            masks = np.stack(masks_list, axis=-1)               # (H, W, C) without torch

        return image, masks

    # def __getitem__(self, index):
    #     # load image 
    #     img_path = os.path.join(self.image_dir, self.images[index])
    #     image = np.array(Image.open(img_path).convert("RGB"))

    #     #  load masks for each class 
    #     masks = []
    #     img_name = self.images[index]
    #     for class_dir in self.class_folder_dir:
    #         mask_path = class_dir / img_name  # assumes same filename as in imgs
    #         if mask_path.exists():
    #             mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    #             mask[mask > 0] = 1.0  # binarize
    #         else:
    #             # if a mask is missing for this class, fill with zeros
    #             mask = np.zeros(image.shape[:2], dtype=np.float32)
    #             print("No masks detected...")
                
    #         masks.append(mask)

    #     # stack into H x W x C
    #     masks = np.stack(masks, axis=-1)

    #     #  optional transforms 
    #     if self.transform is not None:
    #         augmented = self.transform(image=image, mask=masks)
    #         image = augmented["image"]
    #         masks = augmented["mask"]
            
    #     return image, mask
