import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]


class FineTuningIntegrationTest(unittest.TestCase):
    def test_fake_data_end_to_end(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "finetune.py"),
                    "--fake-data",
                    "--output-dir",
                    output_dir,
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("fake-data smoke test passed", result.stdout)
            self.assertTrue((Path(output_dir) / "training.pt").is_file())
            self.assertTrue((Path(output_dir) / "inference.pt").is_file())

    @unittest.skipUnless(
        os.environ.get("NITROGEN_CHECKPOINT"),
        "Set NITROGEN_CHECKPOINT to run against the released checkpoint",
    )
    def test_released_checkpoint_loads_for_finetuning(self):
        script = """
import os
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from nitrogen.finetuning.checkpoint import load_pretrained_components
from nitrogen.finetuning.data import FineTuningCollator, ManifestFineTuningDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, processor, config, mapping = load_pretrained_components(
    os.environ["NITROGEN_CHECKPOINT"], device
)
assert any(parameter.requires_grad for parameter in model.parameters())
tokenizer.train()
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    image = directory / "frame.png"
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(image)
    horizon = config.model_cfg.action_horizon
    buttons = config.model_cfg.action_dim - 4
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"samples": [{
        "frames": [image.name] * config.modality_cfg.frame_per_sample,
        "buttons": [[0] * buttons for _ in range(horizon)],
        "j_left": [[0, 0] for _ in range(horizon)],
        "j_right": [[0, 0] for _ in range(horizon)],
        "game": None
    }]}))
    dataset = ManifestFineTuningDataset(manifest, processor)
    loader = DataLoader(
        dataset, batch_size=1, collate_fn=FineTuningCollator(tokenizer)
    )
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    loss = model(batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
print(config.experiment_name)
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
