from pathlib import Path

import matplotlib
import torch
from config_helper.config import dumb_load_yaml_to_omegaconf, save_config
from liso.slim.experiment import Experiment
from liso.utils.config_helper_helper import load_handle_args_cfg_logdir, parse_cli_args

matplotlib.use("Agg")


def main():
    assert torch.cuda.is_available(), "CPU only not supported"
    log_dir_prefix = None

    args = parse_cli_args()
    if args.override_summary_dir:
        args, cfg, maybe_slow_log_dir = load_handle_args_cfg_logdir(
            args=args,
            save_cfg=not args.inference_only,
            log_dir=args.summary_dir,
            log_dir_prefix=log_dir_prefix,
        )
    else:
        args, cfg, maybe_slow_log_dir = load_handle_args_cfg_logdir(
            args=args, save_cfg=not args.inference_only, log_dir_prefix=log_dir_prefix
        )

    log_dir = maybe_slow_log_dir

    if args.inference_only:
        assert args.load_checkpoint is not None
        cfg_path_chkpt = Path(args.load_checkpoint).parent.parent.joinpath("config.yml")

        checkpoint_cfg = dumb_load_yaml_to_omegaconf(cfg_path_chkpt)
        checkpoint_cfg.data.paths = cfg.data.paths  # data paths may have changed
        save_config(checkpoint_cfg, log_dir.joinpath("config.yml"))
        del cfg
        exp = Experiment(
            cfg=checkpoint_cfg,
            slim_cfg=checkpoint_cfg.SLIM,
            log_dir=log_dir,
            maybe_slow_log_dir=maybe_slow_log_dir,
            debug=args.fast_test,
            world_size=args.world_size,
            worker_id=args.worker_id,
        )
        exp.prepare(for_training=False)
        exp.load_model_weights(args.load_checkpoint)
        exp.run_inference_only(skip_existing=args.override_summary_dir)
    else:
        exp = Experiment(
            cfg=cfg,
            slim_cfg=cfg.SLIM,
            log_dir=log_dir,
            maybe_slow_log_dir=maybe_slow_log_dir,
            debug=args.fast_test,
        )
        exp.prepare(for_training=True)
        exp.run()


if __name__ == "__main__":
    main()
