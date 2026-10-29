from data import utils
from model.losses import LISNetLoss
from utilities import logging
from functionalities import train
from model import serialization


def run(device, config, gp=0.75):
    """Run finetuning on a pre-trained LISNet model.
    
    Loads pre-trained weights, freezes backbone (except final stage), and trains
    on new dataset with reduced learning rates.
    
    Args:
        device: PyTorch device (cuda or cpu).
        config (dict): Configuration dictionary with training parameters and checkpoint path.
    """
    lr_factor = config.get("finetune_lr_factor", 0.2)
    state = "finetune"
    train_loader, val_loader, test_loader, num_classes, _ = utils.fetch_data(
        config.get("root"),
        config.get("num_workers"),
        config.get("augment"),
        config.get("batch_size"),
        config.get("subset"),
        config.get("classes"),
        config.get("data_caching"),
    )
    logging.prog_state(f"DATA LOADED SUCCESSFULLY | TRAIN: {len(train_loader.dataset)} | VAL: {len(val_loader.dataset)} | TEST: {len(test_loader.dataset)} | CLASSES: {num_classes}")
    criterion = LISNetLoss(
        config.get("w_center", 1.5),
        config.get("w_offset", 1.0),
        config.get("w_mask", 5.0),
        config.get("w_score", 2.0),
        config.get("w_semantic", 1.0),
    ).to(device)
    model = serialization.load_for_finetuning(criterion, config.get("weights"), device)
    optimizer, scheduler = serialization.build_optimizer_and_scheduler(
        model, 
        state, 
        config.get("lr"), 
        config.get("max_epochs"),
        lr_factor,
        config.get("cosine_factor"),
    )
    train.train_loop(
        state,
        model, 
        criterion, 
        train_loader, 
        val_loader,
        test_loader,
        scheduler, 
        optimizer, 
        device, 
        config,
        gp,
    )