import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from nitrogen.mm_tokenizers import NitrogenTokenizer


class ManifestFineTuningDataset(Dataset):
    """Load pre-windowed frame/action demonstrations from a JSON manifest."""

    def __init__(self, manifest_path: str | Path, image_processor: Any):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        self.samples = manifest["samples"]
        if not self.samples:
            raise ValueError("The fine-tuning manifest contains no samples")
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        frame_paths = sample["frames"]
        images = []
        for frame_path in frame_paths:
            path = Path(frame_path)
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        frames = self.image_processor(images, return_tensors="pt")["pixel_values"]
        action_horizon = len(sample["buttons"])
        if not (
            action_horizon == len(sample["j_left"]) == len(sample["j_right"])
        ):
            raise ValueError(f"Action lengths differ in sample {index}")
        if any(len(buttons) != 21 for buttons in sample["buttons"]):
            raise ValueError(f"Expected 21 buttons per timestep in sample {index}")
        if any(len(stick) != 2 for stick in sample["j_left"] + sample["j_right"]):
            raise ValueError(f"Expected two coordinates per joystick in sample {index}")
        if not all(
            -1 <= coordinate <= 1
            for stick in sample["j_left"] + sample["j_right"]
            for coordinate in stick
        ):
            raise ValueError(f"Joystick value outside [-1, 1] in sample {index}")

        dropped_frames = sample.get("dropped_frames", [False] * len(images))
        if len(dropped_frames) != len(images):
            raise ValueError(f"dropped_frames length differs from frames in sample {index}")

        return {
            "frames": frames,
            "dropped_frames": np.asarray(dropped_frames, dtype=bool),
            "buttons": np.asarray(sample["buttons"], dtype=np.float32)[None, ...],
            "j_left": np.asarray(sample["j_left"], dtype=np.float32)[None, ...],
            "j_right": np.asarray(sample["j_right"], dtype=np.float32)[None, ...],
            "game": sample.get("game"),
        }


class FineTuningCollator:
    def __init__(self, tokenizer: NitrogenTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        encoded = [self.tokenizer.encode(sample) for sample in samples]
        keys = (
            "images",
            "dropped_images",
            "actions",
            "actions_mask",
            "has_real_action",
            "vl_token_ids",
            "sa_token_ids",
            "vl_attn_mask",
            "embodiment_id",
            "game_ids",
        )
        batch = {}
        for key in keys:
            values = [
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                for value in (item[key] for item in encoded)
            ]
            batch[key] = torch.stack(values)

        batch["images"] = batch["images"].float()
        batch["actions"] = batch["actions"].float()
        batch["actions_mask"] = batch["actions_mask"].bool()
        batch["has_real_action"] = batch["has_real_action"].bool()
        batch["dropped_images"] = batch["dropped_images"].bool()
        batch["vl_attn_mask"] = batch["vl_attn_mask"].bool()
        for key in ("vl_token_ids", "sa_token_ids", "embodiment_id", "game_ids"):
            batch[key] = batch[key].long()
        return batch
