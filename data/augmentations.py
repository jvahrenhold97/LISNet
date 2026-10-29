import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def _resize_targets(instance, semantic, out_stride):
    """Resize instance and semantic maps by downsampling factor.
    
    Args:
        instance (np.ndarray): Instance segmentation map (H, W).
        semantic (np.ndarray): Semantic segmentation map (H, W).
        out_stride (int): Downsampling factor (e.g., 2 for 2x reduction).
    
    Returns:
        tuple: (resized_instance, resized_semantic) using nearest neighbor interpolation.
    """
    if out_stride == 1:
        return instance, semantic
    H, W = instance.shape
    Hs, Ws = H // out_stride, W // out_stride
    return (
        cv2.resize(instance, (Ws, Hs), interpolation=cv2.INTER_NEAREST),
        cv2.resize(semantic, (Ws, Hs), interpolation=cv2.INTER_NEAREST),
    )


def get_train_augs(image_size, ds_mean, ds_std):
    """Build training augmentation pipeline with geometric, photometric, and noise transforms.

    Args:
        image_size (int): Target image size.
        ds_mean (list): Dataset mean for normalization [R, G, B].
        ds_std (list): Dataset std for normalization [R, G, B].

    Returns:
        albumentations.Compose: Augmentation pipeline.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.3, 1.0),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(
            limit=20,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.SomeOf(
            transforms=[
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                ]),
                A.GaussNoise(std_range=(0.1, 0.5), p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=3),
                ]),
                A.CLAHE(clip_limit=2.0),
                A.Sharpen(alpha=(0.2, 0.4)),
            ],
            n=2,
            p=1.0,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(image_size // 32, image_size // 16),
            hole_width_range=(image_size // 32, image_size // 16),
            fill=0,
            fill_mask=None,
            p=0.3,
        ),
        A.Normalize(mean=ds_mean, std=ds_std),
        ToTensorV2(),
    ])


def get_val_augs(image_size, ds_mean, ds_std):
    """Build validation/test augmentation pipeline with only resize and normalization.

    Args:
        image_size (int): Target image size.
        ds_mean (list): Dataset mean for normalization [R, G, B].
        ds_std (list): Dataset std for normalization [R, G, B].

    Returns:
        albumentations.Compose: Augmentation pipeline.
    """
    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.Normalize(
            mean=ds_mean,
            std=ds_std,
        ),
        ToTensorV2(),
    ])