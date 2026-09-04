from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoImageProcessor

from nitrogen.cfg import CkptConfig
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.mm_tokenizers import NitrogenTokenizer


def load_pretrained_components(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[NitroGen, NitrogenTokenizer, Any, CkptConfig, dict | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    game_mapping = checkpoint.get("game_mapping")
    tokenizer = NitrogenTokenizer(
        ckpt_config.tokenizer_cfg,
        game_mapping=game_mapping,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        ckpt_config.model_cfg.vision_encoder_name
    )
    model = NitroGen(ckpt_config.model_cfg, game_mapping=tokenizer.game_mapping)
    model.load_state_dict(checkpoint["model"], strict=True)
    tokenizer.train()
    return model.to(device), tokenizer, image_processor, ckpt_config, tokenizer.game_mapping


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_config: CkptConfig,
    game_mapping: dict | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "ckpt_config": ckpt_config.model_dump(),
            "game_mapping": game_mapping,
            "rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng_state"])
    return int(checkpoint["step"])


def export_inference_checkpoint(
    path: str | Path,
    model: nn.Module,
    ckpt_config: CkptConfig,
    game_mapping: dict | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ckpt_config": ckpt_config.model_dump(),
            "game_mapping": game_mapping,
        },
        path,
    )
