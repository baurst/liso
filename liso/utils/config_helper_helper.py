import argparse
import functools as ft
import json
import random
import sys
from datetime import datetime
from inspect import getsourcefile
from pathlib import Path

import numpy as np
import torch
from config_helper.config import (
    dumb_load_yaml_to_omegaconf,
    get_config_hash,
    parse_config,
    save_config,
)
from omegaconf import OmegaConf


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_cli_args(default_cfg="liso_config.yml"):
    default_cfg_file = get_config_dir() / Path(default_cfg)

    parser = argparse.ArgumentParser(
        description="Pipeline to train a NN model specified by a YML config"
    )
    parser.add_argument("--fast-test", action="store_true", help="run super fast test")
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="run small training for profiling using cProfile",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--worker_id",
        default=0,
        type=int,
        help="id of the worker process",
    )
    parser.add_argument(
        "--force_use_initial_bev_range",
        action="store_true",
        help="only relevant for zero-shot generalization methods, will use initial bev range for tracking",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="will load checkpoint and finetune from there, but not optimizer state/training config",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run small training for torch profiler profiling",
    )
    parser.add_argument(
        "--inference-only", action="store_true", help="run inference only"
    )
    parser.add_argument(
        "--override-summary-dir",
        action="store_true",
        help="don't modify summary-dir at all",
    )
    parser.add_argument("--summary-dir", type=str, help="Where to write summaries to.")
    parser.add_argument(
        "--eval_predictions_pkl",
        type=str,
        help="Path to pickled predictions to be evaluated.",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, help="load model weights from this path"
    )
    parser.add_argument("--keys_value", "-kv", action="append", nargs="+", default=[])
    parser.add_argument(
        "-cf",
        "--config_file",
        default=default_cfg_file,
    )
    parser.add_argument(
        "--configs", "-c", action="append", nargs="*", type=str, default=[]
    )
    parser.add_argument(
        "--export_predictions_to_dir",
        type=str,
        help="will save predictions into this directory",
    )
    parser.add_argument(
        "--export_predictions_for_tcr",
        action="store_true",
        help="will trigger usage of kitti_tracking",
    )
    parser.add_argument(
        "--export_predictions_for_visu",
        type=str,
        help="will export some detections for visualization (blender)",
    )
    parser.add_argument(
        "--dump_sequences_for_visu",
        action="store_true",
        help="will export whole (tracked) sequences for visualization (blender)",
    )
    parser.add_argument(
        "--override_network",
        type=str,
        choices=[
            "echo_gt",
            "flow_cluster_detector",
        ],
        help="will override the used network (tracking/eval only!)",
    )

    args = parser.parse_args()

    assert (
        args.world_size > args.worker_id
    ), f"world_size {args.world_size} must be greater than worker_id {args.worker_id}"

    args.configs = ft.reduce(lambda a, b: a + b, [[]] + args.configs)

    all_cli_args_as_str = " ".join(sys.argv[1:])
    print("launch.json style arguments:")
    print(('",\n"').join(all_cli_args_as_str.split(" ")))
    return args


def load_handle_args_cfg_logdir(
    args=None,
    default_cfg="liso_config.yml",
    log_dir=None,
    create_log_dir=True,
    save_cfg=True,
    load_incremental_config=True,
    log_dir_prefix=None,
):
    if args is None:
        args = parse_cli_args(default_cfg)
    if load_incremental_config:
        cfg = parse_config(
            args.config_file,
            extra_cfg_args=args.configs,
            key_value_updates=args.keys_value,
        )
    else:
        assert len(args.configs) == 0, "dumb yaml loading will ignore your extra args"
        assert (
            len(args.keys_value) == 0
        ), "dumb yaml loading will ignore your extra args"
        cfg = dumb_load_yaml_to_omegaconf(args.config_file)
    if log_dir is None:
        cfg_hash = get_config_hash(cfg)[:5]
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"{args.summary_dir}/{cfg_hash}").joinpath(f"{start_time_str}")
    else:
        log_dir = Path(log_dir)
    if log_dir_prefix is not None:
        log_dir = Path(log_dir_prefix).joinpath(log_dir)
    if create_log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to {log_dir}")
    if save_cfg:
        save_config(cfg, log_dir.joinpath("config.yml"))

    set_seed(cfg.seed)
    torch.autograd.set_detect_anomaly(cfg.set_detect_anomaly)
    return args, cfg, log_dir


def get_config_dir():
    return Path(getsourcefile(lambda: 0)).parent.parent / Path("config")


def pretty_json(hp):
    if OmegaConf.is_dict(hp):
        json_hp = "{0}".format(OmegaConf.to_yaml(hp))
    else:
        json_hp = json.dumps(hp, indent=2)

    return "".join("\t" + line for line in json_hp.splitlines(True))
