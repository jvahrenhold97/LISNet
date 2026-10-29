from utilities import logging
import subprocess


def run(device, config):
    """Run ablation studies with different model configurations.
    
    Executes multiple training experiments with different architectural components
    disabled/enabled (backbone size, spatial attention, regularization, etc.).
    
    Args:
        device: PyTorch device (cuda or cpu).
        config (dict): Base configuration dictionary with dataset root.
    """
    experiments = [
        {
            "name": "01_convnext_tiny",
            "flags": ["--tiny_backbone", "--disable_coordatt", "--disable_dropblock"],
            "mode": "--train", "max_epochs": 30, "augmentations": 0,
        },
        {
            "name": "02_baseline",
            "flags": ["--disable_coordatt", "--disable_dropblock"],
            "mode": "--train", "max_epochs": 30, "augmentations": 0,
        },
        {
            "name": "03_drop_block",
            "flags": ["--disable_coordatt"],
            "mode": "--train", "max_epochs": 30, "augmentations": 0,
        },
        {
            "name": "04_coord_att",
            "flags": [],
            "mode": "--train", "max_epochs": 30, "augmentations": 0,
        },
        {
            "name": "05_spatial_bias",
            "flags": ["--enable_spatial_bias"],
            "mode": "--train", "max_epochs": 30, "augmentations": 0,
        },
        {
            "name": "06_ms_dilation",
            "flags": ["--enable_ms_dilation"],
            "mode": "--train", "max_epochs": 30, "augmentations": 0,
        },
        {
            "name": "07_closing",
            "flags": ["--enable_closing"],
            "mode": "--train", "max_epochs": 30, "augmentations": 0,
        },
        {
            "name": "08_tiny_full",
            "flags": ["--tiny_backbone"],
            "mode": "--train", "max_epochs": 150, "augmentations": 3,
        },
        {
            "name": "01PLUS_convnext_tiny",
            "flags": ["--tiny_backbone", "--disable_coordatt", "--disable_dropblock"],
            "mode": "--train", "max_epochs": 150, "augmentations": 3,
        },
        {
            "name": "02PLUS_baseline",
            "flags": ["--disable_coordatt", "--disable_dropblock"],
            "mode": "--train", "max_epochs": 150, "augmentations": 3,
        },
        {
            "name": "03PLUS_drop_block",
            "flags": ["--disable_coordatt"],
            "mode": "--train", "max_epochs": 150, "augmentations": 3,
        },
        {
            "name": "04PLUS_coord_att",
            "flags": [],
            "mode": "--train", "max_epochs": 150, "augmentations": 3,
        },
    ]
    for exp in experiments:
        cmd = [
            "python", "main.py", 
            exp["mode"], 
            "--root", config.get("root"), 
            "--run_name", exp["name"],
            "--max_epochs", str(exp["max_epochs"]),
            "--augment", str(exp["augmentations"])
        ] + exp["flags"]
        logging.prog_state(f"Running experiment '{cmd}'.")
        subprocess.run(cmd, check=True) 