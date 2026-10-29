import random
import torch
from torch.utils.data import DataLoader, Subset
from data.dataset import InstanceDataset, infer_dataset_image_size, assess_dataset


def collate_batch(samples: list[dict]) -> dict:
    """Collate sample dictionaries into batched tensors."""
    if "centers" not in samples[0]:
        batch_images = [s["image"] for s in samples]
        batch_meta = [s.get("meta", {}) for s in samples]
        return {
            "image": torch.stack(batch_images, dim=0),
            "meta": batch_meta,
        }
    batch_images = []
    batch_semantic = []
    batch_visibility = []
    batch_centers = []
    batch_center_dist = []
    batch_offsets = []
    batch_offset_mask = []
    batch_instance = []
    batch_ambiguous_s2 = []
    batch_ambiguous_s4 = []
    batch_meta = []
    for s in samples:
        batch_images.append(s["image"])
        batch_centers.append(s["centers"])
        batch_offsets.append(s["offsets"])
        batch_center_dist.append(s["center_dist"])
        batch_ambiguous_s2.append(s["valid_spatial_mask_s2"])
        batch_ambiguous_s4.append(s["valid_spatial_mask_s4"])
        batch_semantic.append(s["semantic"])
        vis = s["visibility"]
        if vis.ndim == 2:
            vis = vis.unsqueeze(0)
        batch_visibility.append(vis)
        batch_offset_mask.append(s["offset_mask"].float())
        if "instance" in s:
            batch_instance.append(s["instance"])
        else:
            batch_instance.append(None)
        batch_meta.append(s.get("meta", {}))
    batch = {
        "image": torch.stack(batch_images, dim=0),
        "centers": torch.stack(batch_centers, dim=0),
        "center_dist": torch.stack(batch_center_dist, dim=0),
        "semantic": torch.stack(batch_semantic, dim=0),
        "visibility": torch.stack(batch_visibility, dim=0),
        "offsets": torch.stack(batch_offsets, dim=0),
        "offset_mask": torch.stack(batch_offset_mask, dim=0),
        "valid_spatial_mask_s2": torch.stack(batch_ambiguous_s2, dim=0),
        "valid_spatial_mask_s4": torch.stack(batch_ambiguous_s4, dim=0),
        "meta": batch_meta,
    }
    if all(isinstance(x, torch.Tensor) for x in batch_instance):
        batch["instance"] = torch.stack(batch_instance, dim=0)
    return batch


def get_dataset(
    root: str,
    augmentations: int,
    split: str,
    cache_dataset: bool = False,
    **kwargs,
) -> InstanceDataset:
    """Create an InstanceDataset instance for a specific split with caching control."""
    return InstanceDataset(
        root_dir=root,
        split=split,
        num_augs=augmentations,
        cache_dataset=cache_dataset,
        **kwargs,
    )


def fetch_data(
    root: str,
    num_workers: int,
    augmentations: int,
    batch_size: int,
    subset_ratio: float = 1.0,
    num_classes: int = 2,
    cache_dataset: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Construct and return train, validation, and test loaders along with class count."""
    dataset_type, _ = assess_dataset(root)
    image_size = infer_dataset_image_size(root, dataset_type)
    train_ds = get_dataset(root, augmentations, "train", cache_dataset=cache_dataset)
    val_ds = get_dataset(root, 0, "val", cache_dataset=cache_dataset)
    test_ds = get_dataset(root, 0, "test", cache_dataset=cache_dataset)
    if subset_ratio < 1.0:
        def get_subset(ds: InstanceDataset) -> Subset:
            """Create a deterministic sample index subset from an input dataset."""
            n = max(1, int(len(ds) * subset_ratio))
            indices = random.Random(1337).sample(range(len(ds)), n)
            return Subset(ds, indices)
        train_ds = get_subset(train_ds)
        val_ds = get_subset(val_ds)
        test_ds = get_subset(test_ds)
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader, test_loader, num_classes, image_size