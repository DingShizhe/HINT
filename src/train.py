from pathlib import Path

import torch
from torch.backends import cudnn
import configargparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.DeepRegression import Model
from src.data.TFR_data import TFRDataModule
from src.data.Mathit_data import MathitDataModule
from src.data.Perlin_data import PerlinDataModule
from src.data.PTV_data import PTVDataModule


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
    elif hparams.dataset_type in ["PTV"]:
        data = PTVDataModule(
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
    elif hparams.dataset_type in ["Perlin"]:
        data = PerlinDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers
        )
    else:
        raise NotImplementedError

    model = Model(
        hparams,
        DefaultMunch.fromDict(eval(hparams.model_arch_cfg))
    )

    print( pl.core.memory.ModelSummary(model, mode="full") )


    # ------------------------
    # 2 INIT TRAINER
    # ------------------------


    # assert hparams.dataset_type in ["TFR", "TFR_FINETUNE"]
    assert hparams.dataset_type in ["Mathit", "TFR", "PTV", "Perlin"]


    # if hparams.wandb_project:
    wandb_logger = TensorBoardLogger(
        hparams.wandb_project,
        # name=hparams.model_name
        name=hparams.checkpoint_dir.split("/")[-1]
    )


    checkpoint_callback = ModelCheckpoint(
        monitor="train/training_mae_step",
        # monitor="val/val_mae",
        # monitor="val/val_mae_niert",
        dirpath="CKPTS/" + hparams.checkpoint_dir,
        filename="log_"+"-{epoch:03d}-{loss:.5f}",
        # mode="min",
        save_top_k=20
    )

    import json

    import os
    os.makedirs("CKPTS/" + hparams.checkpoint_dir, exist_ok=True)

    with open("CKPTS/" + hparams.checkpoint_dir + "/hparams.json", "w") as f:
        json.dump(vars(hparams), f, indent=4)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # print(hparams.resume_from_checkpoint)

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpu,
        precision=16 if hparams.use_16bit else 32,
        val_check_interval=hparams.val_check_interval,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        profiler=hparams.profiler,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=hparams.gradient_clip
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    print(hparams)
    print()
    # trainer.fit(model, data)
    trainer.fit(model, data)

    # trainer.test()
