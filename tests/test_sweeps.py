import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_pest_detection_experiments(tmp_path):
    """Test running pest detection model configurations."""
    command = [
        startfile,
        "-m",
        "experiment=pest_detection",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
        "+dataset.pest_classes=['aphids', 'beetles', 'caterpillars', 'mites', 'thrips', 'whiteflies']",
        "+model.backbone=resnet50",
        "+dataset.image_size=224",
        "+dataset.num_workers=4",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_pest_detection_sweep(tmp_path):
    """Test hyperparameter sweep for pest detection model."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.backbone=resnet50,efficientnet_b0,mobilenet_v3",
        "model.optimizer.lr=0.001,0.0001",
        "model.batch_size=16,32,64",
        "model.dropout=0.3,0.5",
        "+model.weight_decay=1e-4,1e-5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_pest_detection_optuna(tmp_path):
    """Test optuna optimization for pest detection."""
    command = [
        startfile,
        "-m",
        "hparams_search=pest_detection_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "+model.augmentation.random_crop=true",
        "+model.augmentation.random_flip=true",
        "+model.augmentation.random_rotation=true",
        "+model.augmentation.color_jitter=true",
        "+model.augmentation.gaussian_blur=true",
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_pest_detection_wandb(tmp_path):
    """Test pest detection with Weights & Biases logging."""
    command = [
        startfile,
        "-m",
        "experiment=pest_detection",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer.max_epochs=10",
        "+trainer.limit_train_batches=0.2",
        "+trainer.limit_val_batches=0.2",
        "+model.backbone=efficientnet_b0",
        "+dataset.image_size=224",
        "+dataset.augmentation=strong",
        "+model.learning_rate_scheduler=cosine",
        "+model.warmup_epochs=2",
        "logger=wandb",
        '+logger.project="pest-detection"',
        '+logger.tags=["experiment", "optimization"]',
    ]
    run_sh_command(command)


@RunIf(sh=True)
def test_pest_detection_inference(tmp_path):
    """Test inference pipeline for pest detection."""
    command = [
        startfile,
        "--config-name=inference",
        "hydra.sweep.dir=" + str(tmp_path),
        "+inference.checkpoint_path=best_model.ckpt",
        "+inference.input_size=224",
        "+inference.confidence_threshold=0.5",
        "+inference.batch_size=32",
        "+inference.use_tta=true",  # Test Time Augmentation
        "+inference.export_format=['onnx', 'torchscript']",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_pest_detection_ensemble(tmp_path):
    """Test ensemble model training and inference."""
    command = [
        startfile,
        "-m",
        "experiment=pest_detection_ensemble",
        "hydra.sweep.dir=" + str(tmp_path),
        "+model.ensemble_models=['resnet50', 'efficientnet_b0', 'mobilenet_v3']",
        "+model.ensemble_method=weighted_average",
        "+model.weights=[0.4, 0.4, 0.2]",
        "+dataset.image_size=224",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_pest_detection_cross_validation(tmp_path):
    """Test k-fold cross validation for pest detection."""
    command = [
        startfile,
        "-m",
        "experiment=pest_detection_cv",
        "hydra.sweep.dir=" + str(tmp_path),
        "+cv.num_folds=5",
        "+cv.stratified=true",
        "+model.backbone=resnet50",
        "+dataset.image_size=224",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)
