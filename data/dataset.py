import os
import cv2
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from data.targets import generate_center_targets
from data.augmentations import get_train_augs, get_val_augs, _resize_targets


_RESOLVE_CACHE = {}


def get_resolved_dataset_dir(root: str, dataset_type: str, split: str, data_type: str) -> str:
    """Resolve directory path for a specific split and data modality with caching."""
    cache_key = (root, dataset_type, split, data_type)
    if cache_key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[cache_key]
    if dataset_type == "PhenoBench":
        mapping = {
            "images": "images",
            "masks": "leaf_instances",
            "semantics": "semantics",
            "leaf_visibility": "leaf_visibility",
        }
        folder_target = mapping.get(data_type, data_type)
        primary = os.path.join(root, split, folder_target)
        if not os.path.isdir(primary) and data_type == "masks":
            primary = os.path.join(root, split, "masks")
    else:
        primary = os.path.join(root, split, data_type)
        if not os.path.isdir(primary):
            aliases = {
                "masks": ["leaf_instances", "instance_masks", "instances"],
                "leaf_visibility": ["visibility", "vis"],
                "semantics": ["semantic", "sem"],
            }
            for alt_name in aliases.get(data_type, []):
                candidate = os.path.join(root, split, alt_name)
                if os.path.isdir(candidate):
                    primary = candidate
                    break
    if not os.path.isdir(primary):
        alt = os.path.join(root, data_type, split)
        if os.path.isdir(alt):
            primary = alt
    _RESOLVE_CACHE[cache_key] = primary
    return primary


def assess_dataset(root: str, **kwargs) -> tuple[str, bool]:
    """Detect whether dataset is PhenoBench or Custom, and check for a labeled test split."""
    if not os.path.exists(root):
        raise FileNotFoundError(f"Root dataset directory does not exist: {root}")
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    splits = ["train", "val", "test"]
    pb_indicators = {
        "semantics",
        "leaf_instances",
        "plant_instances",
        "leaf_visibility",
    }

    def has_files(folder_path: str) -> bool:
        """Check if target directory exists and contains valid image files."""
        if not os.path.isdir(folder_path):
            return False
        return any(f.lower().endswith(valid_exts) for f in os.listdir(folder_path))

    is_phenobench = False
    for split in splits:
        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            entries = set(os.listdir(split_dir))
            if entries & pb_indicators:
                is_phenobench = True
                break
    if is_phenobench:
        test_dir = os.path.join(root, "test")
        has_test = os.path.isdir(test_dir) and (
            has_files(os.path.join(test_dir, "leaf_instances"))
            or has_files(os.path.join(test_dir, "masks"))
        )
        return "PhenoBench", has_test
    has_custom_splits = any(
        os.path.isdir(os.path.join(root, s, "images"))
        and (
            os.path.isdir(os.path.join(root, s, "masks"))
            or os.path.isdir(os.path.join(root, s, "leaf_instances"))
        )
        for s in splits
    )
    has_custom_root = os.path.isdir(os.path.join(root, "images")) and (
        os.path.isdir(os.path.join(root, "masks"))
        or os.path.isdir(os.path.join(root, "leaf_instances"))
    )
    if has_custom_splits or has_custom_root:
        test_img_dir = os.path.join(root, "test", "images")
        test_mask_dir = os.path.join(root, "test", "masks")
        if not os.path.isdir(test_mask_dir):
            test_mask_dir = os.path.join(root, "test", "leaf_instances")
        has_test = has_files(test_img_dir) and has_files(test_mask_dir)
        return "Custom", has_test
    raise ValueError(
        f"Directory structure in '{root}' does not match either PhenoBench "
        "or Custom format."
    )


def infer_dataset_image_size(root_dir: str, dataset_type: str, max_samples: int = 30) -> int:
    """Sample dataset image resolutions to identify the mode dimension rounded to multiples of 32."""
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    dims = []
    for split in ["train", "val", "test"]:
        img_dir = get_resolved_dataset_dir(root_dir, dataset_type, split, "images")
        if not os.path.isdir(img_dir):
            continue
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        for fname in files[:max_samples]:
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                dims.append(max(h, w))
        if len(dims) >= max_samples:
            break
    if not dims:
        return 512
    most_common_dim = Counter(dims).most_common(1)[0][0]
    aligned_dim = int(round(most_common_dim / 32.0) * 32)
    return max(64, aligned_dim)


def load_and_convert_sample(root: str, split: str, img_name: str, dataset_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sample image and convert masks into standardized instance, semantic, and visibility targets."""
    img_dir = get_resolved_dataset_dir(root, dataset_type, split, "images")
    img_path = os.path.join(img_dir, img_name)
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image at: {img_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    if dataset_type == "PhenoBench":
        mask_dir = get_resolved_dataset_dir(root, dataset_type, split, "masks")
        sem_dir = get_resolved_dataset_dir(root, dataset_type, split, "semantics")
        vis_dir = get_resolved_dataset_dir(root, dataset_type, split, "leaf_visibility")
        inst_raw = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_UNCHANGED)
        sem_raw = cv2.imread(os.path.join(sem_dir, img_name), cv2.IMREAD_UNCHANGED)
        if inst_raw is None or sem_raw is None:
            raise FileNotFoundError(f"Missing PhenoBench ground truth for {img_name}")
        sem_raw = sem_raw.astype(np.int32)
        inst_raw = inst_raw.astype(np.int32)
        is_leaf = sem_raw == 1
        is_void = (sem_raw == 255) | (sem_raw == 3) | (sem_raw == 4)
        semantic = np.zeros((h, w), dtype=np.int32)
        semantic[is_leaf] = 1
        semantic[is_void] = 255
        instance = np.zeros((h, w), dtype=np.int32)
        instance[is_leaf] = inst_raw[is_leaf]
        vis_path = os.path.join(vis_dir, img_name)
        vis_raw = cv2.imread(vis_path, cv2.IMREAD_UNCHANGED)
        visibility = np.zeros((h, w), dtype=np.float32)
        if vis_raw is not None:
            if vis_raw.ndim > 2:
                vis_raw = vis_raw[:, :, 0]
            vis_norm = (
                (vis_raw.astype(np.float32) / 255.0)
                if vis_raw.dtype == np.uint8 and vis_raw.max() > 1
                else vis_raw.astype(np.float32)
            )
            visibility[is_leaf] = vis_norm[is_leaf]
        else:
            visibility[is_leaf] = 1.0
    else:
        mask_dir = get_resolved_dataset_dir(root, dataset_type, split, "masks")
        stem = os.path.splitext(img_name)[0]
        candidates = [
            os.path.join(mask_dir, f"{stem}_mask.png"),
            os.path.join(mask_dir, img_name),
            os.path.join(mask_dir, f"{stem}.png"),
        ]
        mask_path = next((p for p in candidates if os.path.exists(p)), None)
        if mask_path is None:
            raise FileNotFoundError(f"Mask file not found for {img_name} in {mask_dir}")
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise ValueError(f"OpenCV failed to read mask at: {mask_path}")
        if raw_mask.ndim > 2:
            raw_mask = raw_mask.astype(np.int32)
            raw_id = raw_mask[:, :, 0] + (raw_mask[:, :, 1] << 8) + (raw_mask[:, :, 2] << 16)
        else:
            raw_id = raw_mask.astype(np.int32)
        is_leaf = (raw_id >= 1) & (raw_id <= 254)
        is_void = raw_id == 255
        instance = np.zeros((h, w), dtype=np.int32)
        instance[is_leaf] = raw_id[is_leaf]
        sem_dir = get_resolved_dataset_dir(root, dataset_type, split, "semantics")
        sem_path = os.path.join(sem_dir, img_name) if os.path.isdir(sem_dir) else None
        if sem_path and os.path.exists(sem_path):
            sem_raw = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
            if sem_raw.ndim > 2:
                sem_raw = sem_raw[:, :, 0]
            semantic = sem_raw.astype(np.int32)
        else:
            semantic = np.zeros((h, w), dtype=np.int32)
            semantic[is_leaf] = 1
            semantic[is_void] = 255
        vis_dir = get_resolved_dataset_dir(root, dataset_type, split, "leaf_visibility")
        vis_path = os.path.join(vis_dir, img_name) if os.path.isdir(vis_dir) else None
        visibility = np.zeros((h, w), dtype=np.float32)
        if vis_path and os.path.exists(vis_path):
            vis_raw = cv2.imread(vis_path, cv2.IMREAD_UNCHANGED)
            if vis_raw.ndim > 2:
                vis_raw = vis_raw[:, :, 0]
            vis_norm = (
                (vis_raw.astype(np.float32) / 255.0)
                if vis_raw.dtype == np.uint8 and vis_raw.max() > 1
                else vis_raw.astype(np.float32)
            )
            visibility[is_leaf] = vis_norm[is_leaf]
        else:
            visibility[is_leaf] = 1.0
    return image, instance, semantic, visibility


class InstanceDataset(Dataset):
    """Instance segmentation dataset supporting optional .npz caching for targets and source modalities."""
    VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_augs: int = 1,
        val_test_split: float = 0.4,
        cache_dataset: bool = False,
        **kwargs,
    ) -> None:
        """Initialize split paths, infer image dimension if unprovided, and configure caching mode."""
        self.root_dir = root_dir
        self.split = split
        self.cache_dataset = cache_dataset
        self.kwargs = kwargs
        self.num_copies = max(1, num_augs) if (num_augs != 0 and split == "train") else 1
        self.dataset_type, self.dataset_has_test_set = assess_dataset(root_dir)
        self.image_size = infer_dataset_image_size(root_dir, self.dataset_type)
        train_img_dir = get_resolved_dataset_dir(root_dir, self.dataset_type, "train", "images")
        val_img_dir = get_resolved_dataset_dir(root_dir, self.dataset_type, "val", "images")
        if self.dataset_has_test_set:
            test_img_dir = get_resolved_dataset_dir(root_dir, self.dataset_type, "test", "images")
            if split == "train":
                self.active_split_path = "train"
                img_list = self._list_images(train_img_dir)
            elif split == "val":
                self.active_split_path = "val"
                img_list = self._list_images(val_img_dir)
            else:
                self.active_split_path = "test"
                img_list = self._list_images(test_img_dir)
        else:
            if split == "train":
                self.active_split_path = "train"
                img_list = self._list_images(train_img_dir)
            else:
                self.active_split_path = "val"
                all_val = self._list_images(val_img_dir)
                all_val.sort()
                split_idx = int(len(all_val) * (1.0 - val_test_split))
                if split == "val":
                    img_list = all_val[:split_idx]
                else:
                    img_list = all_val[split_idx:]
        self.images = [{"file_name": f} for f in img_list]
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.is_augmented = (num_augs > 0 and split == "train")
        if self.is_augmented:
            self.transform = get_train_augs(self.image_size, mean, std)
        else:
            self.transform = get_val_augs(self.image_size, mean, std)

        if self.cache_dataset:
            cache_folder_name = f".cache_{self.split}_{self.image_size}"
            self.cache_dir = os.path.join(self.root_dir, cache_folder_name)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None

    def _list_images(self, path: str) -> list[str]:
        """Fetch sorted image file names matching supported image formats."""
        if not os.path.isdir(path):
            return []
        return sorted([f for f in os.listdir(path) if f.lower().endswith(self.VALID_EXTS)])

    def __len__(self) -> int:
        """Return dataset size scaled by augmentation copies."""
        return len(self.images) * self.num_copies

    def _load_or_create_cache(self, img_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve preprocessed base sample arrays from disk cache or create the cache entry."""
        if not self.cache_dir:
            return load_and_convert_sample(self.root_dir, self.active_split_path, img_name, self.dataset_type)
        
        base_cache_file = os.path.join(self.cache_dir, f"{os.path.splitext(img_name)[0]}_base.npz")
        if os.path.isfile(base_cache_file):
            try:
                data = np.load(base_cache_file)
                return data["image"], data["instance"], data["semantic"], data["visibility"]
            except Exception:
                pass

        image, instance, semantic, visibility = load_and_convert_sample(
            self.root_dir,
            self.active_split_path,
            img_name,
            self.dataset_type,
        )
        try:
            np.savez_compressed(
                base_cache_file,
                image=image,
                instance=instance,
                semantic=semantic,
                visibility=visibility,
            )
        except Exception:
            pass
        return image, instance, semantic, visibility

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        """Load native sample, apply standardized spatial scaling, and compute stride targets."""
        img_meta = self.images[idx % len(self.images)]
        img_name = img_meta["file_name"]

        if self.cache_dir and not self.is_augmented:
            target_cache_file = os.path.join(self.cache_dir, f"{os.path.splitext(img_name)[0]}_targets.npz")
            if os.path.isfile(target_cache_file):
                try:
                    c = np.load(target_cache_file)
                    return {
                        "image": torch.from_numpy(c["image"]),
                        "instance": torch.from_numpy(c["instance"]).long(),
                        "semantic": torch.from_numpy(c["semantic"]).long(),
                        "visibility": torch.from_numpy(c["visibility"]).float(),
                        "centers": torch.from_numpy(c["centers"]).float(),
                        "center_dist": torch.from_numpy(c["center_dist"]).float(),
                        "offsets": torch.from_numpy(c["offsets"]).float(),
                        "offset_mask": torch.from_numpy(c["offset_mask"]).float(),
                        "valid_spatial_mask_s2": torch.from_numpy(c["valid_spatial_mask_s2"]).float(),
                        "valid_spatial_mask_s4": torch.from_numpy(c["valid_spatial_mask_s4"]).float(),
                        "meta": {"image_id": img_name},
                    }
                except Exception:
                    pass

        image, instance, semantic, visibility = self._load_or_create_cache(img_name)
        augmented = self.transform(
            image=image,
            mask=instance,
            masks=[semantic.astype(np.int32), visibility],
        )
        image_t = augmented["image"]
        instance_aug = (
            augmented["mask"].long().numpy()
            if hasattr(augmented["mask"], "numpy")
            else np.array(augmented["mask"], dtype=np.int64)
        )
        semantic_aug = (
            augmented["masks"][0].numpy().astype(np.int64)
            if hasattr(augmented["masks"][0], "numpy")
            else np.asarray(augmented["masks"][0], dtype=np.int64)
        )
        visibility_aug = (
            augmented["masks"][1].numpy().astype(np.float32)
            if hasattr(augmented["masks"][1], "numpy")
            else np.asarray(augmented["masks"][1], dtype=np.float32)
        )
        inst_s2, sem_s2 = _resize_targets(instance_aug, semantic_aug, 2)
        vis_s2 = cv2.resize(
            visibility_aug,
            (self.image_size // 2, self.image_size // 2),
            interpolation=cv2.INTER_NEAREST,
        )
        geo_targets = generate_center_targets(
            instance_aug,
            target_stride_geo=4.0,
            target_stride_sem=2.0,
            sigma_min=0.5,
        )
        sem_s4 = cv2.resize(
            semantic_aug.astype(np.uint8),
            (self.image_size // 4, self.image_size // 4),
            interpolation=cv2.INTER_NEAREST,
        )
        valid_mask_s4 = geo_targets["valid_spatial_mask_s4"]
        valid_mask_s4[0, sem_s4 == 255] = 0.0
        valid_mask_s2 = geo_targets["valid_spatial_mask_s2"]
        valid_mask_s2[0, sem_s2 == 255] = 0.0

        if self.cache_dir and not self.is_augmented:
            try:
                np.savez_compressed(
                    target_cache_file,
                    image=image_t.cpu().numpy(),
                    instance=inst_s2,
                    semantic=sem_s2,
                    visibility=vis_s2,
                    centers=geo_targets["centers"].cpu().numpy(),
                    center_dist=geo_targets["center_dist"].cpu().numpy(),
                    offsets=geo_targets["offsets"].cpu().numpy(),
                    offset_mask=geo_targets["offset_mask"].cpu().numpy(),
                    valid_spatial_mask_s2=valid_mask_s2.cpu().numpy(),
                    valid_spatial_mask_s4=valid_mask_s4.cpu().numpy(),
                )
            except Exception:
                pass

        return {
            "image": image_t,
            "instance": torch.from_numpy(inst_s2),
            "semantic": torch.from_numpy(sem_s2).long(),
            "visibility": torch.from_numpy(vis_s2).float(),
            "centers": geo_targets["centers"],
            "center_dist": geo_targets["center_dist"],
            "offsets": geo_targets["offsets"],
            "offset_mask": geo_targets["offset_mask"],
            "valid_spatial_mask_s2": valid_mask_s2,
            "valid_spatial_mask_s4": valid_mask_s4,
            "meta": {"image_id": img_name},
        }