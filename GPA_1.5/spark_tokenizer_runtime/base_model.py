# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:28
# Author    :Hui Huang
import json

import torch
import torch.nn as nn
import yaml

from .tokenizer_utils import load_config
import os
from safetensors.torch import load_file


class SparkBaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, model_path: str):
        config = load_config(os.path.join(model_path, "config.yaml"))['audio_tokenizer']
        model = cls(config)
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.remove_weight_norm()
        return model

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


class SnacBaseModel(nn.Module):
    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, model_path: str):
        model = cls.from_config(os.path.join(model_path, "config.json"))
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model


class MegaBaseModel(nn.Module):
    CKPT_NAME = "model"

    @classmethod
    def from_pretrained(cls, model_path: str):
        config_file = None
        ckpt_path = None
        for file in os.listdir(model_path):
            if file.endswith(".ckpt"):
                ckpt_path = os.path.join(model_path, file)
            if file.endswith(".yaml"):
                config_file = os.path.join(model_path, file)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict_all = {
            k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in checkpoint["state_dict"].items()
        }
        state_dict = state_dict_all[cls.CKPT_NAME]
        state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        if config_file is not None:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            model = cls(config)
        else:
            model = cls()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model