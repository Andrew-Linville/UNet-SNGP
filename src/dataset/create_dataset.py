from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_coco_json(json_path):
    
    print("Loaded annotations")
    coco = COCO(json_path)
    


    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images.")
    
    return coco, img_ids

def train_val_shuffle(img_ids, train_ratio=0.8, return_numpy=False):
    """
    Inputs: img_ids
    """
    seed = 42
    img_ids = np.array(img_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(img_ids))

    cut = int(len(img_ids) * train_ratio)
    train_ids = img_ids[perm[:cut]]
    val_ids   = img_ids[perm[cut:]]
    
    if return_numpy:
        return train_ids, val_ids
    else: 
        return train_ids.tolist(), val_ids.tolist()
    

# Working!!
def create_masks(coco, img_ids, out_folder=None, save_mask=False):
    """ I believe this function will only return a single mask; no mutli channel masks for different classes

    Args:
        coco (_type_): _description_
        img_ids (_type_): _description_
        out_folder (_type_): _description_
        save_mask (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_
    """
    
    total_mask_list = []
    for img_id in img_ids:
        mask_list = []
        for class_name in CLASS_NAMES: 
            
            cat_ids = coco.getCatIds(catNms=[class_name])
            info = coco.loadImgs(img_id)[0]
            H, W = info["height"], info["width"]
            combined_mask = np.zeros((H, W), dtype=bool)

    
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
            for ann in coco.loadAnns(ann_ids):
                combined_mask |= coco.annToMask(ann).astype(bool)   # union of instances
                # mask = coco.annToMask(ann)
                # combined_mask = np.maximum(combined_mask,mask)
                
                
            scaled_mask = (combined_mask.astype(np.uint8)*MASK_VALUE).astype(np.uint8)
            
            if save_mask:
                # ? PIL Saving
                suffix = class_name.replace(" ", "_")
                stem = Path(info.get("file_name", f"img_{img_id}")).stem
                out_path = Path(OUT_DIR)/ out_folder / suffix / f"{stem}__{suffix}.png" #! <---- Need to set up data file path accordingly
                # print(np.max(scaled_mask))
                # print(scaled_mask.shape)
                # print(type(scaled_mask))
                # Image.fromarray(combined_mask.astype(np.uint8) * MASK_VALUE, mode="L").save(out_path)
                Image.fromarray(scaled_mask, mode="L").save(out_path)
            
            # ? Matplotlib Saving
            # stem = Path(info.get("file_name", f"img_{img_id}")).stem
            # out_path = Path(OUT_DIR) / out_folder / f"{stem}__{suffix}.png"
            # scaled_mask = combined_mask.astype(np.uint8) * MASK_VALUE
            # # Save using matplotlib
            # plt.imsave(out_path, scaled_mask, cmap="gray")
            
            mask_list.append(scaled_mask)
            # plt.figure()
            # plt.imshow(scaled_mask)
            # plt.show()
        # Stack the lists?
        total_mask_list.append(mask_list)
    return total_mask_list

def save_imgs_to_folder(coco, img_ids, output_folder):
    out_dir = Path(OUT_DIR) / output_folder

    saved = 0
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        file_name = info["file_name"]

        src = Path(IMAGES_DIR) / file_name
        dst = out_dir / Path(file_name).name

        if not src.exists():
            print(f"[warn] Source image not found: {src}")
            continue

        img = Image.open(src).convert("RGB")
        img.save(dst)  # requires that `out_dir` already exists
        saved += 1

    print(f"Saved {saved} images → {out_dir}")

def chunk_image(img, chunk_size=512, stride=0):
    
    arr = np.asarray(img)
    H, W = arr.shape[:2]
    if stride is None or stride <= 0:
        stride = chunk_size

    def positions(size, win, step):
        # If the image dimension is smaller than the window, just start at 0
        if size <= win:
            return [0]
        # Regular stepping
        pos = list(range(0, size - win + 1, step))
        # Ensure the very last window ends at the edge (collision)
        last_needed = size - win
        if pos[-1] != last_needed:
            pos.append(last_needed)
        # Deduplicate while preserving order (handles step that divides evenly)
        seen, out = set(), []
        for p in pos:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

    ys = positions(H, chunk_size, stride)
    xs = positions(W, chunk_size, stride)

    chunks = []
    for y in ys:
        for x in xs:
            # Numpy slicing naturally clips at image borders; since we never
            # let y/x exceed size - chunk_size (unless image smaller than window),
            # chunks are either exact `chunk_size` or the full image if smaller.
            chunk = arr[y:y+chunk_size, x:x+chunk_size, ...]
            chunks.append(chunk)

    return chunks 

def load_images(coco, img_ids):
    img_list = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        file_name = info["file_name"]
        src = Path(IMAGES_DIR) / file_name 
        img = Image.open(src).convert("RGB")
        img_list.append(img)
    return img_list



def sub_sample_images(coco, img_list, mask_list, img_ids):
    output_img_chunks = []
    output_mask_chunks = []
    output_img_ids = []

    for img, masks, img_id in zip(img_list, mask_list, img_ids):
        img_chunks = chunk_image(img)  # -> list[np.ndarray], length = num_chunks

        # For each mask in this image, get its list of chunks
        mask_chunks_list = [chunk_image(mask) for mask in masks]  # list[list[np.ndarray]]
        # Transpose so we iterate per-chunk: [(m1_chunk0, m2_chunk0, ...), (m1_chunk1, ...), ...]
        mask_chunks_per_chunk = list(zip(*mask_chunks_list))

        info = coco.loadImgs(img_id)[0]
        file_name = info["file_name"]

        for idx, (img_chunk, mask_chunk_tuple) in enumerate(zip(img_chunks, mask_chunks_per_chunk)):
            # mask_chunk_tuple is a tuple of np.ndarrays (one per class) for this chunk
            # Keep chunk only if any mask in this chunk has a nonzero pixel
            if any(np.max(m) != 0 for m in mask_chunk_tuple):
                output_img_chunks.append(img_chunk)
                # store as a list (or tuple) of arrays corresponding to classes for this chunk
                output_mask_chunks.append(list(mask_chunk_tuple))
                output_img_ids.append(f"{file_name}_{idx}")

    return output_img_chunks, output_mask_chunks, output_img_ids


def save_chunks_to_folder(img_chunks, mask_chunks, id_chunks, img_folder, mask_folder):
    saved = 0
    out_dir_img = Path(OUT_DIR) / img_folder
    out_dir_mask = Path(OUT_DIR) / mask_folder
    out_dir_img.mkdir(parents=True, exist_ok=True)
    out_dir_mask.mkdir(parents=True, exist_ok=True)

    for img_arr, mask_arr, file_id in zip(img_chunks, mask_chunks, id_chunks):
        
        # Ensure numpy arrays
        img_arr = np.asarray(img_arr)
        pil_img = Image.fromarray(img_arr.astype(np.uint8), mode="RGB")
        dst_img = out_dir_img / f"{file_id}.png"
        
        
        for mask, class_name in zip(mask_arr, CLASS_NAMES):
            mask_arr = np.asarray(mask)
            pil_mask = Image.fromarray(mask_arr.astype(np.uint8), mode="L")
            # dst_mask = out_dir_mask / class_name /f"{file_id}_{class_name}.png"
            dst_mask = out_dir_mask / class_name /f"{file_id}.png"

            pil_mask.save(dst_mask)
            
        # Save
        pil_img.save(dst_img)
        
        saved += 1

    print(f"Saved {saved} image/mask pairs → {out_dir_img}, {out_dir_mask}")

def delete_all_files_in_folder(folder_path: str):
    """
    Deletes all files within a specified folder (non-recursive).
    Subdirectories and their contents are not affected.
    """
    # Convert string to Path and resolve relative to ROOT
    folder = (ROOT / Path(folder_path)).resolve()

    if not folder.exists():
        print(f"Error: Folder '{folder}' not found.")
        return

    deleted = 0
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
            print(f"Deleted file: {item}")
            deleted += 1

    print(f"Deleted {deleted} files in '{folder}'.")


CLASS_NAMES = ["particle", "dark_spot"]
ROOT = Path(__file__).resolve().parent.parent  # go up to project root
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
    
    if CLEAR_DIRS:
        folders_to_clear = [train_output_imgs_folder, train_output_mask_folder, val_output_mask_folder, val_output_imgs_folder]
        for folder in folders_to_clear:
            fixed_folder_path = f"data/{folder}" # <- need to resolve root of problem
            delete_all_files_in_folder(fixed_folder_path)
            
    if CREATE_DATASET:    
        coco, img_ids= load_coco_json(JSON_PATH)
        train_ids, val_ids = train_val_shuffle(img_ids)
        
        if not SUBSAMPLE_IMGS:
            create_masks(coco, val_ids, val_output_mask_folder, True)
            create_masks(coco, train_ids, train_output_mask_folder, True)
            
            save_imgs_to_folder(coco, val_ids, val_output_imgs_folder)
            save_imgs_to_folder(coco, train_ids, train_output_imgs_folder)
            
        else:
            mask_list_val = create_masks(coco, val_ids, val_output_mask_folder, False)
            mask_list_train = create_masks(coco, train_ids, train_output_mask_folder, False)
            
            img_list_val = load_images(coco, val_ids)
            img_list_train = load_images(coco, train_ids)
            
            img_chunks_val, mask_chunks_val, id_chunks_val = sub_sample_images(coco, img_list_val, mask_list_val, val_ids)
            img_chunks_train, mask_chunks_train, id_chunks_train = sub_sample_images(coco, img_list_train, mask_list_train, train_ids)
            
            save_chunks_to_folder(img_chunks_val, mask_chunks_val, id_chunks_val, val_output_imgs_folder, val_output_mask_folder)
            save_chunks_to_folder(img_chunks_train, mask_chunks_train, id_chunks_train, train_output_imgs_folder, train_output_mask_folder)
            
if __name__ == "__main__":        
    main()