import argparse, torch, os
from collections import defaultdict


from data import dataset
from utilities import cfgparse, logging
from functionalities import train, finetune, inference, label, ablate, test


torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.serialization.add_safe_globals([defaultdict])
torch.serialization.add_safe_globals([list])


def get_args():
    parser = argparse.ArgumentParser(description="LISNet Training & Ablation Pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--finetune", action="store_true")
    mode.add_argument("--ablate", action="store_true")
    mode.add_argument("--inference", action="store_true")
    mode.add_argument("--label", action="store_true")
    mode.add_argument("--test", action="store_true")
    hardware = parser.add_argument_group('Hardware & Data')
    hardware.add_argument("--root", type=str, required=True)
    hardware.add_argument("--run_name", type=str, default="lisnet")
    hardware.add_argument("--batch_size", type=int, default=4)
    hardware.add_argument("--num_workers", type=int, default=8)
    hardware.add_argument("--inference_window_size", type=int, default=512)
    hardware.add_argument("--data_caching", type=bool, default=True)
    train_params = parser.add_argument_group('Training Params')
    train_params.add_argument("--lr", type=float, default=5e-5)
    train_params.add_argument("--subset", type=float, default=1.0)
    train_params.add_argument("--max_epochs", type=int, default=150)
    train_params.add_argument("--patience", type=int, default=15)
    train_params.add_argument("--augment", type=int, default=1)
    train_params.add_argument("--classes", type=int, default=2)
    ablate_params = parser.add_argument_group('Ablation Config')
    ablate_params.add_argument('--tiny_backbone', action='store_true')
    ablate_params.add_argument('--enable_closing', action='store_true')
    ablate_params.add_argument('--enable_ms_dilation', action='store_true')
    ablate_params.add_argument('--disable_coordatt', action='store_true')
    ablate_params.add_argument('--disable_dropblock', action='store_true')
    ablate_params.add_argument('--enable_spatial_bias', action='store_true')
    parser.add_argument("--config", type=str, default="model/model_config.yaml")
    parser.add_argument("--weights", type=str, default="dataset_lisnet/--MOD--/lisnet.pt", help="Path to model checkpoint.")
    args = parser.parse_args()
    if args.inference:
        if args.weights is None:
            parser.error("--inference requires --weights 'Path/to/Weights/'")
    if args.finetune:
        if args.weights is None:
            parser.error("--finetune requires --weights 'Path/to/Weights/'")
    if args.label:
        if args.weights is None:
            parser.error("--label requires --weights 'Path/to/Weights/'")
    if args.test:
        if args.weights is None:
            parser.error("--test requires --weights 'Path/to/Weights/'")
    return args


def get_dynamic_run_name(args, dataset_name):
    base_name = args.run_name if args.run_name else "lisnet"
    full_name = f"{dataset_name}_{base_name}"
    return full_name


def main():
    cli_args = get_args()
    model_cfg_raw = cfgparse.parse_config(cli_args.config)
    model_cfg_flat = {}
    for key, value in model_cfg_raw.items():
        if isinstance(value, dict):
            model_cfg_flat.update(value)
        else:
            model_cfg_flat[key] = value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_config = {**model_cfg_flat, **vars(cli_args)}
    if cli_args.label == True:
        logging.prog_state("Commencing LISNet pseudo-labeling...")
        label.run(device, full_config)
    elif cli_args.inference == True:
        logging.prog_state("Running LISNet inference...")
        inference.run(device, full_config)
    else:
        dataset_name = os.path.basename(cli_args.root.strip("/"))
        run_id = get_dynamic_run_name(cli_args, dataset_name)
        logging.log_configuration(cli_args, device, run_id)
        full_config.update({
            "run_id": run_id,
            "device": device,
            "dataset_name": dataset_name
        })
        if cli_args.train == True:
            logging.prog_state("Commencing LISNet training...")
            train.run(device, full_config)
        elif cli_args.finetune == True:
            logging.prog_state("Commencing LISNet finetuning...")
            finetune.run(device, full_config)
        elif cli_args.test == True:
            logging.prog_state("Commencing LISNet testing...")
            test.run_independent(device, full_config)
        elif cli_args.ablate == True:
            logging.prog_state("Commencing LISNet ablation...")
            ablate.run(device, full_config)
        else:
            raise RuntimeError("Unexpected Exception!")




if __name__ == "__main__":
    main()