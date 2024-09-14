import functools
import hashlib
import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def dumb_load_yaml_to_omegaconf(cfg_path: str):
    class PythonicLoader(yaml.SafeLoader):
        pass

    PythonicLoader.add_constructor("!tuple", yaml.FullLoader.construct_python_tuple)

    yaml_pythonic_load = functools.wraps(yaml.load)(
        functools.partial(yaml.load, Loader=PythonicLoader)
    )
    with open(cfg_path, "r") as f:
        file_cfg = OmegaConf.create(yaml_pythonic_load(f))

    return file_cfg


def update_nested_dict(d, other):
    for k, v in other.items():
        d_v = d.get(k)
        if isinstance(v, Mapping) and isinstance(d_v, Mapping):
            update_nested_dict(d_v, v)
        else:
            assert k == "meta_cfgs" or k in d, f"default value for key {k} not found!"
            d[k] = deepcopy(v)


def recursive_cfg_update(full_immutable_cfg, cfg, addon_config_name):
    addon_config = full_immutable_cfg[addon_config_name]
    if "meta_cfgs" in addon_config:
        assert not isinstance(addon_config["meta_cfgs"], str), "string not allowed"

        for meta_cfg_name in addon_config["meta_cfgs"]:
            recursive_cfg_update(full_immutable_cfg, cfg, meta_cfg_name)

    print("Updating using meta_cfg {0}".format(addon_config_name))
    update_nested_dict(cfg, addon_config)


def parse_config(cfg_path, extra_cfg_args=(), key_value_updates=None, verbose=False):
    if key_value_updates is not None:
        assert isinstance(key_value_updates, (tuple, list))
        for entry in key_value_updates:
            assert isinstance(entry, (tuple, list))
    print("Loading config: {0}".format(cfg_path))

    assert isinstance(extra_cfg_args, (list, tuple)), type(extra_cfg_args)

    file_cfg = dumb_load_yaml_to_omegaconf(cfg_path)

    cfg = deepcopy(file_cfg.default)
    if verbose:
        logger = logging.getLogger()
        logger.info("Default config:")
        logger.info(OmegaConf.to_yaml(cfg))
        print("Default config:")
        print(OmegaConf.to_yaml(cfg))
    for extra_cfg in extra_cfg_args:
        recursive_cfg_update(file_cfg, cfg, extra_cfg)
    if verbose:
        logger.info("Intermediate config:")
        logger.info(OmegaConf.to_yaml(cfg))
        print("Intermediate config:")
        print(OmegaConf.to_yaml(cfg))
    if key_value_updates is not None and len(key_value_updates) > 0:
        for kvupd in key_value_updates:
            for el in kvupd:
                assert "=" not in el, f"char = not allowed in keyword update: {el}"
        dotlist = [
            ".".join(kv_upd[:-1]) + "=" + kv_upd[-1] for kv_upd in key_value_updates
        ]
        for entry in dotlist:
            assert (
                OmegaConf.select(cfg, entry.split("=")[0]) is not None
            ), f"you are updating keys that do not exist in default config: {entry}"
        update_subcfg = OmegaConf.from_dotlist(dotlist)
        if verbose:
            logger.info("Updating from key_value_updates with:")
            logger.info(OmegaConf.to_yaml(update_subcfg))
            print("Updating from key_value_updates with:")
            print(OmegaConf.to_yaml(update_subcfg))
        # import ipdb; ipdb.set_trace()
        cfg = OmegaConf.merge(cfg, update_subcfg)
    if verbose:
        logger.info("Final config:")
        logger.info(OmegaConf.to_yaml(cfg))
    print("Final config:")
    print(OmegaConf.to_yaml(cfg))
    return cfg


def save_config(cfg: OmegaConf, path: Path):
    OmegaConf.save(config=cfg, f=path)


def get_config_str(cfg: OmegaConf):
    try:
        return OmegaConf.to_yaml(cfg)
    except ValueError:
        return OmegaConf.to_yaml(OmegaConf.create(dict(cfg)))


def get_config_hash(cfg: OmegaConf):
    cfg_str = get_config_str(cfg)
    hash_object = hashlib.sha256(f"{cfg_str}".encode("utf-8"))
    hex_dig = hash_object.hexdigest()
    return hex_dig
