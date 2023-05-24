"""
Runs a model on a single node across multiple gpus.
"""
from pathlib import Path

import torch
from torch.backends import cudnn
import configargparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.DeepRegression import Model
from src.data.TFR_data import TFRDataModule
from src.data.Mathit_data import MathitDataModule
from src.data.PTV_data import PTVDataModule
from src.data.Perlin_data import PerlinDataModule

from munch import DefaultMunch

import pdb

def main(hparams):

    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

    # ------------------------
    # 1 INIT LIGHTNING MODEL DATA
    # ------------------------

    if hparams.dataset_type in ["TFR"]:
        data = TFRDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers,
        )
    elif hparams.dataset_type == "Mathit":
        data = MathitDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers,
            DefaultMunch.fromDict(eval(hparams.mathit_data_cfg))
        )
    elif hparams.dataset_type == "PTV":
        data = PTVDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers,
        )
    elif hparams.dataset_type == "Perlin":
        data = PerlinDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers,
        )
    else:
        raise NotImplementedError

    data.setup()

    model = Model(
        hparams,
        DefaultMunch.fromDict(eval(hparams.model_arch_cfg)),
        default_layout=data.default_layout if hasattr(data, 'default_layout') else None,
    )

    print( pl.core.memory.ModelSummary(model, mode="full") )

    print(hparams.resume_from_checkpoint)
    print(hparams.resume_from_checkpoint)
    print(hparams.resume_from_checkpoint)
    print(hparams.resume_from_checkpoint)

    # assert hparams.resume_from_checkpoint
    print("Load Pre-Trained Model From", hparams.resume_from_checkpoint, "...")
    checkpoint = torch.load(hparams.resume_from_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------


    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpu,
        precision=16 if hparams.use_16bit else 32,
        profiler=hparams.profiler,
    )


    # ------------------------
    # 3 START TESTING
    # ------------------------
    print(hparams)
    print()
    # trainer.fit(model, data)
    trainer.test(model, data.test_dataloader())
    # trainer.test(model, data.val_dataloader())

    # trainer.test()
