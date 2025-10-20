import numpy as np
import matplotlib.pyplot as plt
from create_dataset import create_masks, load_coco_json, train_val_shuffle, sub_sample_images, load_images
from pathlib import Path


CLASS_NAMES = ["particle", "dark_spot"]
ROOT = Path(__file__).resolve().parent.parent  # go up to project root
print(ROOT)
JSON_PATH = ROOT / "data" / "double_class_particle.json"
IMAGES_DIR = ROOT / "data" / "images"
MASK_VALUE = 255.0
OUT_DIR = ROOT / "data"
SUBSAMPLE_IMGS = True
CLEAR_DIRS = True
CREATE_DATASET = True



def main():  
    
    print("running main")
    # Create the train masks and save train images
    train_output_mask_folder = r"train/masks"
    train_output_imgs_folder = r"train/imgs"
    # Create the val masks and save val images
    val_output_mask_folder = r"val/masks"
    val_output_imgs_folder = r"val/imgs"
    

    if CREATE_DATASET:    
        coco, img_ids= load_coco_json(JSON_PATH)
        train_ids, val_ids = train_val_shuffle(img_ids)

        mask_list = create_masks(coco, val_ids)
        print("Created Masks")


        mask_set_1 = mask_list[0]
        
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(mask_set_1[0])
        # plt.subplot(1,2,2)
        # plt.imshow(mask_set_1[1])
        # plt.show()
        
        img_list = load_images(coco, val_ids)
        print("Loaded images")
        output_img_chunks, output_mask_chunks, output_img_ids = sub_sample_images(coco, img_list, mask_list, img_ids)
        print("Ran Subsample function")

        output_chunk_1 = output_img_chunks[0]
        output_mask_chunk_1 = output_mask_chunks[0]
        print(len(output_mask_chunks))
        print(len(output_mask_chunk_1))
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(output_chunk_1)
        plt.subplot(1,3,2)
        plt.imshow(output_mask_chunk_1[0])
        plt.subplot(1,3,3)
        plt.imshow(output_mask_chunk_1[1])
        plt.show()

if __name__ == "__main__":        
    main()