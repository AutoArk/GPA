# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:27
# Author    :Hui Huang
from omegaconf import OmegaConf, DictConfig
import torch


def load_config(config_path: str) -> DictConfig:
    """Loads a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config


def gpu_supports_fp16() -> bool:
    # 1. 确保 CUDA 可用
    if not torch.cuda.is_available():
        return False

    # 2. 获取设备的 compute capability
    major, minor = torch.cuda.get_device_capability()

    # 3. 判断是否 >= 5.3
    if major > 5 or (major == 5 and minor >= 3):
        return True
    else:
        return False


def get_dtype(device: str):
    if device.startswith('cuda') and gpu_supports_fp16():
        return torch.float16
    else:
        return torch.float32