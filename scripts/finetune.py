import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from nitrogen.cfg import CkptConfig, ModalityConfig
from nitrogen.finetuning.checkpoint import (
    load_pretrained_components,
    load_training_checkpoint,
)
from nitrogen.finetuning.data import FineTuningCollator, ManifestFineTuningDataset
from nitrogen.finetuning.trainer import FineTuningConfig, FineTuningTrainer
from nitrogen.flow_matching_transformer.modules import (
    DiTConfig,
    SelfAttentionTransformerConfig,
)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig


class TinyImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("TinyImageProcessor only supports PyTorch tensors")
        arrays = [
            np.asarray(image.resize((16, 16)), dtype=np.float32).transpose(2, 0, 1)
            / 255.0
            for image in images
        ]
        return {"pixel_values": torch.from_numpy(np.stack(arrays))}


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Conv2d(3, hidden_size, kernel_size=8, stride=8)

    def forward(self, images):
        features = self.projection(images).flatten(2).transpose(1, 2)
        return {"last_hidden_state": features}


def create_fake_manifest(output_dir: Path, action_horizon: int) -> Path:
    data_dir = output_dir / "fake_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    generator = np.random.default_rng(7)
    for sample_index in range(6):
        frame_path = data_dir / f"frame_{sample_index}.png"
        pixels = generator.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(frame_path)
        samples.append(
            {
                "frames": [frame_path.name],
                "buttons": generator.integers(
                    0, 2, size=(action_horizon, 21)
                ).tolist(),
                "j_left": generator.uniform(
                    -1, 1, size=(action_horizon, 2)
                ).tolist(),
                "j_right": generator.uniform(
                    -1, 1, size=(action_horizon, 2)
                ).tolist(),
                "dropped_frames": [False],
                "game": "fake-game",
            }
        )
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest_path


def create_tiny_components(device: torch.device):
    hidden_size = 16
    action_horizon = 4
    action_dim = 25
    model_cfg = NitroGen_Config(
        hidden_size=hidden_size,
        vision_hidden_size=hidden_size,
        action_dim=action_dim,
        action_horizon=action_horizon,
        num_timestep_buckets=16,
        num_inference_timesteps=2,
        vision_encoder_name="offline-tiny",
        tune_vision_tower=False,
        tune_vl_mixing=False,
        diffusion_model_cfg=DiTConfig(
            num_attention_heads=2,
            attention_head_dim=8,
            output_dim=hidden_size,
            num_layers=1,
            dropout=0.0,
            cross_attention_dim=hidden_size,
            max_num_positional_embeddings=action_horizon,
        ),
        vl_self_attention_cfg=SelfAttentionTransformerConfig(
            num_attention_heads=2,
            attention_head_dim=8,
            output_dim=hidden_size,
            num_layers=1,
            dropout=0.0,
            max_num_positional_embeddings=8,
        ),
    )
    tokenizer_cfg = NitrogenTokenizerConfig(
        num_visual_tokens_per_frame=4,
        max_action_dim=action_dim,
        max_sequence_length=8,
        action_horizon=action_horizon,
    )
    ckpt_config = CkptConfig(
        experiment_name="synthetic-smoke-test",
        model_cfg=model_cfg,
        tokenizer_cfg=tokenizer_cfg,
        modality_cfg=ModalityConfig(
            frame_per_sample=1,
            frame_spacing=1,
            action_per_chunk=action_horizon,
        ),
    )
    game_mapping = {None: 0, "fake-game": 1}
    tokenizer = NitrogenTokenizer(tokenizer_cfg, game_mapping=game_mapping)
    model = NitroGen(
        model_cfg,
        game_mapping=game_mapping,
        vision_encoder=TinyVisionEncoder(hidden_size),
    ).to(device)
    return model, tokenizer, TinyImageProcessor(), ckpt_config, game_mapping


def make_loaders(manifest, image_processor, tokenizer):
    dataset = ManifestFineTuningDataset(manifest, image_processor)
    collator = FineTuningCollator(tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, validation_loader


def snapshot_parameters(model, trainable):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad == trainable
    }


def assert_parameter_behavior(model, trainable_before, frozen_before):
    named_parameters = dict(model.named_parameters())
    if not any(
        named_parameters[name].grad is not None for name in trainable_before
    ):
        raise AssertionError("No trainable parameter received a gradient")
    if not any(
        not torch.equal(value, named_parameters[name].detach())
        for name, value in trainable_before.items()
    ):
        raise AssertionError("No trainable parameter was updated")
    changed_frozen = [
        name
        for name, value in frozen_before.items()
        if not torch.equal(value, named_parameters[name].detach())
    ]
    if changed_frozen:
        raise AssertionError(f"Frozen parameters changed: {changed_frozen}")


def run_fake_smoke_test(output_dir: Path, device: torch.device) -> None:
    torch.manual_seed(7)
    model, tokenizer, processor, ckpt_config, game_mapping = create_tiny_components(
        device
    )
    manifest = create_fake_manifest(
        output_dir, ckpt_config.model_cfg.action_horizon
    )
    train_loader, validation_loader = make_loaders(manifest, processor, tokenizer)
    trainer = FineTuningTrainer(
        model,
        train_loader,
        validation_loader,
        FineTuningConfig(output_dir=output_dir, max_steps=2, save_every=2),
        ckpt_config,
        game_mapping,
        device,
    )
    trainable_before = snapshot_parameters(model, trainable=True)
    frozen_before = snapshot_parameters(model, trainable=False)
    losses = trainer.train()
    if len(losses) != 2 or not all(np.isfinite(losses)):
        raise AssertionError(f"Expected two finite losses, got {losses}")
    assert_parameter_behavior(model, trainable_before, frozen_before)
    validation_loss = trainer.validate()
    training_path = trainer.save()

    resumed_model, resumed_tokenizer, resumed_processor, resumed_config, resumed_mapping = (
        create_tiny_components(device)
    )
    resumed_train, resumed_validation = make_loaders(
        manifest, resumed_processor, resumed_tokenizer
    )
    resumed_trainer = FineTuningTrainer(
        resumed_model,
        resumed_train,
        resumed_validation,
        FineTuningConfig(output_dir=output_dir, max_steps=3, save_every=3),
        resumed_config,
        resumed_mapping,
        device,
    )
    resumed_trainer.step = load_training_checkpoint(
        training_path, resumed_model, resumed_trainer.optimizer
    )
    if resumed_trainer.step != 2:
        raise AssertionError("Training checkpoint did not restore step 2")
    resumed_trainer.train()
    resumed_trainer.save()
    inference_path = resumed_trainer.export()

    exported = torch.load(inference_path, map_location="cpu", weights_only=False)
    verification_model, _, _, _, _ = create_tiny_components(torch.device("cpu"))
    verification_model.load_state_dict(exported["model"], strict=True)
    if "ckpt_config" not in exported:
        raise AssertionError("Inference export is missing ckpt_config")
    print(
        f"fake-data smoke test passed: steps=3 "
        f"validation_loss={validation_loss:.6f} export={inference_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NitroGen")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fake_data:
        run_fake_smoke_test(args.output_dir, device)
        return
    if args.checkpoint is None or args.manifest is None:
        raise SystemExit("--checkpoint and --manifest are required without --fake-data")

    model, tokenizer, processor, ckpt_config, game_mapping = (
        load_pretrained_components(args.checkpoint, device)
    )
    model.set_trainable_parameters(
        tune_vision_tower=False,
        tune_vl_mixing=False,
        tune_diffusion_model=True,
        tune_multi_projector=True,
    )
    dataset = ManifestFineTuningDataset(args.manifest, processor)
    collator = FineTuningCollator(tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    trainer = FineTuningTrainer(
        model,
        train_loader,
        validation_loader,
        FineTuningConfig(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
        ),
        ckpt_config,
        game_mapping,
        device,
    )
    if args.resume:
        trainer.step = load_training_checkpoint(
            args.resume, model, trainer.optimizer
        )
    trainer.train()
    validation_loss = trainer.validate()
    trainer.save()
    export_path = trainer.export()
    print(f"validation_loss={validation_loss:.6f} export={export_path}")


if __name__ == "__main__":
    main()
