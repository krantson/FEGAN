import h5py
import matplotlib.image as mpimg
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np
import glob


def crop_patch(img, img_size=(512, 512), patch_size=(150, 150), stride=150, random_crop=False):
    count = 0
    patch_list = []
    if random_crop == True:
        crop_num = 100
        pos = [(np.random.randint(patch_size, img_size[0] - patch_size), np.random.randint(patch_size, img_size[1] - patch_size))
               for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
               range(patch_size[0], img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list


def gen_dataset(src_files, dst_path, crop=False):
    h5py_name = dst_path
    h5f = h5py.File(h5py_name, 'w')

    for i in tqdm(range(len(src_files))):
        # print(src_files[i])
        img = mpimg.imread(src_files[i]) # (h,w,c) and RGB
        [h, w, c] = img.shape
        if crop:
            patch_list = crop_patch(img, (h, w), (150, 150), 150, False)
        else:
            patch_list = [img]

        for num in range(len(patch_list)):
            data = patch_list[num].copy()
            key = str(i) + '_' + str(num) if crop else os.path.splitext(os.path.split(src_files[i])[-1])[0]
            shape = (300, 300, 3) if crop else img.shape
            h5f.create_dataset(key, shape=shape, data=data)
    h5f.close()


def gen_clean_noisy_dataset(noisy_src_path="./dataset/train_noisy/", clean_src_path="./dataset/train_clean/"):
    path = Path(noisy_src_path)
    noisy_dst_path = path.parent / Path(path.parts[-1] + '.h5')

    if os.path.exists(noisy_dst_path):
        print(f"{noisy_dst_path} has been already exsits.")
    else:
        noisy_src_files = glob.glob(os.path.join(noisy_src_path , "*.[jp][pn]g"))
        print("start noisy set...")
        gen_dataset(noisy_src_files, noisy_dst_path, crop=False)
        print("end noisy set...")

    path = Path(clean_src_path)
    clean_dst_path = path.parent / Path(path.parts[-1] + '.h5')

    if os.path.exists(clean_dst_path):
        print(f"{clean_dst_path} has been already exsits.")
    else:
        clean_src_files = glob.glob(os.path.join(clean_src_path , "*.[jp][pn]g"))
        print("start clean set...")
        gen_dataset(clean_src_files, clean_dst_path, crop=False)
        print("end clean set...")
    return noisy_dst_path, clean_dst_path


if __name__ == "__main__":

    gen_clean_noisy_dataset()

