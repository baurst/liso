from inspect import getsourcefile
from pathlib import Path
import unittest

from config_helper.config import parse_config


class TestCase(unittest.TestCase):

   def test_file_exists(self,):
        test_cfg_file = Path(getsourcefile(lambda: 0)).parent / Path("test_config.yml")
        assert test_cfg_file.exists(), self.test_cfg_file
        cfg = parse_config(test_cfg_file)
        assert isinstance(cfg.log_everything, bool), type(cfg.log_everything)
        del cfg

        cfg = parse_config(test_cfg_file, extra_cfg_args=("adam",))
        assert cfg.optimizer.name == "adam", cfg.optimizer.name
        del cfg

        cfg = parse_config(test_cfg_file, extra_cfg_args=("hard_mode",))
        assert cfg.optimizer.name == "adam", cfg.optimizer.name
        assert cfg.data.bev_extent[0] == -40.0, cfg.data.bev_extent
        del cfg

        kv_updates = (
            ("data", "data_dirs", "carla", "/tmp/foo/bar/baz"),
            ("optimizer", "name", "sgd"),
            ("log_everything", "True"),
        )
        cfg = parse_config(
            test_cfg_file, extra_cfg_args=("hard_mode",), key_value_updates=kv_updates
        )

        assert cfg.optimizer.name == "sgd", cfg.optimizer.name
        assert cfg.data.bev_extent[0] == -40.0, cfg.data.bev_extent
        assert isinstance(cfg.log_everything, bool), type(cfg.log_everything)
        assert cfg.log_everything, cfg.log_everything
        del cfg

        cfg = parse_config(
            test_cfg_file,
            extra_cfg_args=("super_hard_mode",),
        )
        assert cfg.optimizer.name == "best_opt", cfg.optimizer.name
        assert cfg.data.bev_extent[0] == -40.0, cfg.data.bev_extent
        assert isinstance(cfg.log_everything, bool), type(cfg.log_everything)
        assert not cfg.log_everything, cfg.log_everything
        assert cfg.random_seed == 333, cfg.random_seed
        assert cfg.optimizer.learning_rate == 0.5, cfg.optimizer.learning_rate
        assert cfg.data.data_dirs.carla == "/some/other/place", cfg.data.data_dirs.carla


