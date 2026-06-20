import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import shutil


def normalize_scan(scan):
    """
    Z-Score 归一化 (对 MRI 非常重要)
    """
    mask = scan > 0
    if mask.sum() == 0:
        return scan
    mean = scan[mask].mean()
    std = scan[mask].std()
    return (scan - mean) / (std + 1e-8)


def process_brats_case(case_dir, output_dir_img, output_dir_mask, target_shape=(128, 128, 128)):
    """
    读取一个病例的4个模态，裁剪并保存为 .npy
    """
    case_id = os.path.basename(case_dir)

    # BraTS 文件命名规则 (根据年份可能微调，这是 2021 标准)
    # e.g., BraTS2021_00000_t1.nii.gz
    modalities = ['t1', 't1ce', 't2', 'flair']
    scans = []

    for mod in modalities:
        file_path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
        if not os.path.exists(file_path):
            print(f"Missing {mod} for {case_id}")
            return
        img = nib.load(file_path).get_fdata().astype(np.float32)
        img = normalize_scan(img)
        scans.append(img)

    # Stack channels: (4, D, H, W)
    image_tensor = np.stack(scans, axis=0)

    # Load Mask
    mask_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
    if os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        # BraTS 原始标签: 0(背景), 1(NCR/NET), 2(ED), 4(ET)
        # 这里将 4 重映射为 3，使得最终标签集合为 {0,1,2,3}
        # 后续 WT/TC/ET 三个指标仍按 BraTS 官方定义聚合:
        # WT = {1,2,3}, TC = {1,3}, ET = {3}
        mask[mask == 4] = 3
    else:
        mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)

    # 简单裁剪：移除周围全黑的背景，居中裁剪，或者 Resize
    # 既然是对标顶刊，我们采用 "Center Crop" 到固定尺寸
    # 实际训练中通常是在 Loader 里随机 Crop，这里我们先保存处理好的大图或者直接 Crop
    # 为了节省磁盘空间，我们这里做一个居中裁剪到 target_shape

    _, D, H, W = image_tensor.shape
    c_d, c_h, c_w = D // 2, H // 2, W // 2
    t_d, t_h, t_w = target_shape

    # 简单的边界检查
    start_d = max(0, c_d - t_d // 2)
    start_h = max(0, c_h - t_h // 2)
    start_w = max(0, c_w - t_w // 2)

    img_crop = image_tensor[:, start_d:start_d + t_d, start_h:start_h + t_h, start_w:start_w + t_w]
    mask_crop = mask[start_d:start_d + t_d, start_h:start_h + t_h, start_w:start_w + t_w]

    # Pad 如果尺寸不够
    if img_crop.shape[1:] != target_shape:
        pad_d = target_shape[0] - img_crop.shape[1]
        pad_h = target_shape[1] - img_crop.shape[2]
        pad_w = target_shape[2] - img_crop.shape[3]
        img_crop = np.pad(img_crop, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
        mask_crop = np.pad(mask_crop, ((0, pad_d), (0, pad_h), (0, pad_w)))

    # 保存
    np.save(os.path.join(output_dir_img, f"{case_id}.npy"), img_crop)
    np.save(os.path.join(output_dir_mask, f"{case_id}.npy"), mask_crop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to BraTS raw data')
    parser.add_argument('--output_root', type=str, default='data_processed', help='Output path')
    args = parser.parse_args()

    # BraTS 结构通常是: Root/BraTS2021_00001/...
    cases = [d for d in os.listdir(args.dataset_root) if os.path.isdir(os.path.join(args.dataset_root, d))]

    out_img = os.path.join(args.output_root, 'train', 'images')
    out_mask = os.path.join(args.output_root, 'train', 'masks')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    print(f"🚀 Processing {len(cases)} cases from BraTS...")

    for case in tqdm(cases):
        case_dir = os.path.join(args.dataset_root, case)
        process_brats_case(case_dir, out_img, out_mask)

    print("✅ Preprocessing Complete!")


if __name__ == "__main__":
    main()