import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from pest_rec.eval import evaluate
from pest_rec.train import train
from pest_rec.models.pest_detector import PestDetector
from pest_rec.data.dataset import PestDataset
from torchvision import transforms


@pytest.mark.slow
def test_train_eval(tmp_path, cfg_train, cfg_eval):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/acc"] > 0.0
    assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001


# Model oluştur
model = PestDetector(num_classes=6, backbone="resnet50")

# Veri dönüşümlerini tanımla
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Dataset oluştur
dataset = PestDataset(
    root_dir="data",
    pest_classes=['aphids', 'beetles', 'caterpillars', 
                 'mites', 'thrips', 'whiteflies'],
    transform=transform
)

print(f"Dataset size: {len(dataset)}")
print(f"Model backbone: {model.backbone_name}")
