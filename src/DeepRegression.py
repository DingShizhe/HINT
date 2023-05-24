from pathlib import Path
from typing import SupportsAbs

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np

import src.models as models
import pdb
from einops import rearrange, repeat

# finetune_mean_factor = (0.34 / 0.094)

def mean_squared_error(orig, pred, mask=None):
    # pdb.set_trace()
    assert len(orig.shape) == 3
    assert len(pred.shape) == 3

    error = (orig - pred) ** 2
    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask[..., None].repeat((1,1,error.size(2)))
        # import pdb; pdb.set_trace()
        error = error * mask
        return error.sum() / mask.sum()
    else:
        return error.sum() / (error.size(0) * error.size(1) * error.size(2))


def mean_squared_error_inter(orig, pred, mask, inter_mask):
    # pdb.set_trace()
    error = (orig - pred) ** 2
    _mask = mask * inter_mask.unsqueeze(-1)
    error = error * _mask
    return error.sum() / _mask.sum()


class Model(LightningModule):
    def __init__(self, hparams, model_arch_cfg, default_layout=None):
        super().__init__()
        self.hparams = hparams
        self.model_arch_cfg = model_arch_cfg
        self._build_model()

        if hparams.dataset_type in ["Mathit"]:
            # self.criterion = nn.MSELoss()
            self.criterion = mean_squared_error
        elif hparams.dataset_type in ["TFR"]:
            self.criterion = nn.L1Loss()
        elif hparams.dataset_type in ["PTV"]:
            self.criterion = mean_squared_error
        elif hparams.dataset_type in ["Perlin"]:
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError

        # if hparams.dataset_type == "TFR_FINETUNE":
        #     self._finetune_mean_factor = finetune_mean_factor
        # else:
        #     self._finetune_mean_factor = 1.0

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.default_layout = default_layout

        # self.mid_points = rearrange(
        #     torch.stack(torch.meshgrid(torch.linspace(-1,1,8), torch.linspace(-1,1,8))),
        #     "d m n -> (m n) d"
        # )


    def _build_model(self):
        model_list = [
            "HINT",
        ]
        self.layout_model = self.hparams.model_name

        assert self.layout_model in model_list, "Error: Model {self.layout_model} Not Defined"


        model_args = dict(
            cfg_dim_input=self.hparams.cfg_dim_input,
            cfg_dim_output=self.hparams.cfg_dim_output,
            n_layers=self.model_arch_cfg.layers,
            d_model=self.model_arch_cfg.dim_hidden,
            d_inner=self.model_arch_cfg.dim_inner,
            n_head=self.model_arch_cfg.num_heads,
            K_0_inv=self.model_arch_cfg.K_0_inv,
            K_min_inv=self.model_arch_cfg.K_min_inv,
            K_min=self.model_arch_cfg.K_min,
            n_blocks=self.model_arch_cfg.n_blocks,
        )

        self.model = getattr(models, self.layout_model) (**model_args)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # if self.hparams.lr_decay < 0.0:
        if self.hparams.dataset_type == "Mathit":
            return optimizer
        elif self.hparams.dataset_type == "PTV":
            # return optimizer
            scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)
            return [optimizer], [scheduler]
        elif self.hparams.dataset_type == "TFR_Scattered":
            scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)
            return [optimizer], [scheduler]
        elif self.hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:
            scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)
            return [optimizer], [scheduler]
        elif self.hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:
            return optimizer
        elif self.hparams.dataset_type == "Mathit_42":
            return optimizer
        elif self.hparams.dataset_type == "Current":
            return optimizer
        elif self.hparams.dataset_type == "Perlin":
            return optimizer
        else:
            raise NotImplementedError


    def forward(self, x):

        masks, features = None, None

        if self.hparams.dataset_type in ["Mathit_42", "PhysioNet", "PhysioNet_FINETUNE"]:
            if self.hparams.dataset_type == "PhysioNet_FINETUNE":
                x = x.clone()
                # Data difference betteen TFR and Mathit
                x[:,:,-1] = (x[:,:,-1]  / 0.268) - 1.0
            output, label, masks = self.model( x )

        elif self.hparams.dataset_type in ["Mathit", "TFR", "TFR_FINETUNE", "Current", "Perlin", "TFR_Scattered", "PTV"]:

            if False:
                RRR = 32

                output, label = self.model( x[0], x[1], x[2], x[3] )
                output_0, label_0 = self.model( x[0], x[1], x[2][:,0:RRR,:], x[3][:,0:RRR,:] )

            else:
                oxs, oys, txs, tys = x[0], x[1], x[2], x[3]

                if self.hparams.observed_as_target:
                    # txs = txs.squeeze(0)
                    # oxs = oxs.squeeze(0)
                    txs = torch.cat([txs, oxs], axis=1)
                    # tys = tys.squeeze(0)
                    # oys = oys.squeeze(0)
                    tys = torch.cat([tys, oys], axis=1)

                if len(x) == 5:                 # mask
                    ret = self.model( oxs, oys, txs, tys, x[4] )
                else:
                    ret = self.model( oxs, oys, txs, tys )


                if len(ret) == 3:
                    output, label, features = ret
                else:
                    output, label = ret
                    features = None

        else:
            raise NotImplementedError

        return output, label, {"masks": masks, "features": features}


    def training_step(self, batch, batch_idx):

        if self.layout_model == "NIERT_PhysioNet":
            heat_pred, heat_label, masks = self(batch)
        else:

            masks = None

            if len(batch) == 5:
                obs_index, heat_obs, pred_index, heat, _ = batch
            elif len(batch) == 6:
                obs_index, heat_obs, pred_index, heat, _, masks = batch
            else:
                assert False

            # heat_info = [obs_index, heat_obs, pred_index, heat]
            heat_info = [obs_index, heat_obs, pred_index, heat, masks]
            heat_pred, heat_label, extra = self(heat_info)

            extra["masks"] = masks

        # self.log("train/batch_idx", batch_idx)
        # self.log("train/heat_label", heat_label.mean() / self._finetune_mean_factor)
        # self.log("train/heat_pred", heat_pred.mean() / self._finetune_mean_factor)

        if self.layout_model in ["NIERT", "NIERTPP"]:

            observed_num = obs_index.size(1)

            if self.hparams.observed_as_target:
                assert False
                masked_loss = self.criterion(                       # objective
                    heat_label[:, observed_num:-observed_num, :],
                    heat_pred[:, observed_num:-observed_num, :],
                    # extra["masks"]                                  # padding masks
                ) * self.hparams.std_heat
                self.log("train/training_mae", masked_loss, on_epoch=True)

                masked_observed_loss = self.criterion(
                    heat_label[:, -observed_num:, :],
                    heat_pred[:, -observed_num:, :]
                ) * self.hparams.std_heat
                self.log("train/training_observed_as_target_mae", masked_observed_loss, on_epoch=True)

            else:

                if extra["masks"] is not None:
                    masked_loss = self.criterion(
                        heat_label[:, observed_num:, :],
                        heat_pred[:, observed_num:, :],
                        extra["masks"]
                    ) * self.hparams.std_heat
                else:
                    masked_loss = self.criterion(
                        heat_label[:, observed_num:, :],
                        heat_pred[:, observed_num:, :]
                    ) * self.hparams.std_heat
                self.log("train/training_mae", masked_loss, on_epoch=True)

            observed_loss = self.criterion(heat_label[:, :observed_num, :], heat_pred[:, :observed_num, :]) * self.hparams.std_heat
            self.log("train/training_observed_mae", observed_loss, on_epoch=True)

            if extra["masks"] is not None:
                __m = torch.ones((heat_label.size(0), observed_num), device=heat_label.device)
                __m = torch.cat([__m, extra["masks"]], axis=1)
                niert_loss = self.criterion(heat_label, heat_pred, __m) * self.hparams.std_heat
            else:
                niert_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat
            self.log("train/training_mae_niert", niert_loss, on_epoch=True)

            LOSS = niert_loss

            return {"loss": LOSS, "masked_loss": masked_loss}

        elif self.layout_model == "HINT":

            assert self.hparams.observed_as_target is True

            if extra["masks"] is not None:
                masked_loss = self.criterion(heat_label, heat_pred, extra["masks"]) * self.hparams.std_heat
            else:
                masked_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat
            
            self.log("train/training_mae", masked_loss, on_epoch=True)

            if extra["features"] is not None:
                LL = len(extra["features"]["o"])
                for ll in range(LL):
                    if ll == 0:
                        observed_loss = self.criterion(extra["features"]["o"][ll], extra["features"]["o_predict"][ll]) * self.hparams.std_heat
                    else:
                        observed_loss += self.criterion(extra["features"]["o"][ll], extra["features"]["o_predict"][ll]) * self.hparams.std_heat
            else:
                observed_loss = 0

            self.log("train/training_observed_mae", observed_loss, on_epoch=True)

            # niert_loss = 0.5 * observed_loss + masked_loss    # PTV
            # niert_loss = 0.5 * observed_loss + masked_loss      # TFR
            # niert_loss = 0.05 * observed_loss + masked_loss      # Mathit
            # niert_loss = 0.01 * observed_loss + masked_loss      # Perlin
            niert_loss = self.hparams.o_loss_coff * observed_loss + masked_loss      # Perlin
            self.log("train/training_mae_niert", niert_loss, on_epoch=True)

            # LLL = len(extra["features"]["o_std_min"])
            # for l in  range(LLL) :
            #     self.log("res_std/o_std_min_%d" % l, extra["features"]["o_std_min"][l], on_epoch=True)
            #     self.log("res_std/o_std_max_%d" % l, extra["features"]["o_std_max"][l], on_epoch=True)


            return {"loss": niert_loss, "masked_loss": masked_loss}


        elif self.layout_model == "NIERT_PhysioNet":
            loss = self.criterion(heat_label, heat_pred, masks[0])
            self.log("train/training_mae", loss, on_epoch=True)

            masked_loss = mean_squared_error_inter(heat_label, heat_pred, masks[0], masks[1])
            self.log("train/training_mae_niert", masked_loss, on_epoch=True)

            return {"loss": loss, "masked_loss": masked_loss}

        else:

            if extra["masks"] is not None:
                loss = self.criterion( heat_label, heat_pred, extra["masks"] ) * self.hparams.std_heat
            else:
                loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat

            self.log("train/training_mae", loss, on_epoch=True)
            return {"loss": loss}


    # def training_epoch_end(self, outputs):
    #     if self.layout_model == "NIERT":
    #         train_loss_mean = torch.stack([x["masked_loss"] for x in outputs]).mean()
    #         self.log("train/train_mae_epoch", train_loss_mean.item())

    #         train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
    #         self.log("train/train_mae_niert_epoch", train_loss_mean.item())
    #     else:
    #         train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
    #         self.log("train/train_mae_epoch", train_loss_mean.item())

    def validation_step(self, batch, batch_idx):

        # pdb.set_trace()

        if self.layout_model == "NIERT_PhysioNet":
            heat_pred, heat_label, masks = self(batch)
        else:

            masks = None
            if len(batch) == 5:
                obs_index, heat_obs, pred_index, heat, _ = batch
            elif len(batch) == 6:
                obs_index, heat_obs, pred_index, heat, _, masks = batch
            else:
                assert False

            heat_info = [obs_index, heat_obs, pred_index, heat, masks]
            heat_pred, heat_label, extra = self(heat_info)

            if masks is not None:
                extra["masks"] = masks

        if self.layout_model in ["NIERT", "NIERTPP"]:

            observed_num = obs_index.size(1)

            if self.hparams.observed_as_target:
                assert False
                masked_loss = self.criterion(
                    heat_label[:, observed_num:-observed_num, :],
                    heat_pred[:, observed_num:-observed_num, :]
                ) * self.hparams.std_heat
                self.log("val/val_mae", masked_loss, on_epoch=True)

                masked_observed_loss = self.criterion(
                    heat_label[:, -observed_num:, :],
                    heat_pred[:, -observed_num:, :]
                ) * self.hparams.std_heat
                self.log("val/val_observed_as_target_mae", masked_observed_loss, on_epoch=True)

            else:
                if extra["masks"] is not None:
                    masked_loss = self.criterion(
                        heat_label[:, observed_num:, :],
                        heat_pred[:, observed_num:, :],
                        extra["masks"]
                    ) * self.hparams.std_heat
                else:
                    masked_loss = self.criterion(
                        heat_label[:, observed_num:, :],
                        heat_pred[:, observed_num:, :]
                    ) * self.hparams.std_heat
                self.log("val/val_mae", masked_loss, on_epoch=True)

            observed_loss = self.criterion(
                heat_label[:, :obs_index.size(1), :],
                heat_pred[:, :obs_index.size(1), :]
            ) * self.hparams.std_heat
            self.log("val/val_observed_mae", observed_loss, on_epoch=True)

            if extra["masks"] is not None:
                __m = torch.ones((heat_label.size(0), observed_num), device=heat_label.device)
                __m = torch.cat([__m, extra["masks"]], axis=1)
                niert_loss = self.criterion(heat_label, heat_pred, __m) * self.hparams.std_heat
            else:
                niert_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat
            self.log("val/val_mae_niert", niert_loss, on_epoch=True)

            LOSS = niert_loss

            return {"val_loss": LOSS, "val_masked_loss": masked_loss}

        elif self.layout_model == "HINT":

            assert self.hparams.observed_as_target is True

            if extra["masks"] is not None:
                masked_loss = self.criterion(heat_label, heat_pred, extra["masks"]) * self.hparams.std_heat
            else:
                masked_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat

            self.log("val/val_mae", masked_loss, on_epoch=True)

            if extra["features"] is not None:
                LL = len(extra["features"]["o"])
                for ll in range(LL):
                    if ll == 0:
                        observed_loss = self.criterion(extra["features"]["o"][ll], extra["features"]["o_predict"][ll]) * self.hparams.std_heat
                    else:
                        observed_loss += self.criterion(extra["features"]["o"][ll], extra["features"]["o_predict"][ll]) * self.hparams.std_heat
            else:
                observed_loss = 0

            self.log("val/val_observed_mae", observed_loss, on_epoch=True)

            # niert_loss = 0.5 * observed_loss + masked_loss    # PTV
            # niert_loss = 0.5 * observed_loss + masked_loss      # TFR
            # niert_loss = 0.05 * observed_loss + masked_loss      # Mathit
            # niert_loss = 0.01 * observed_loss + masked_loss      # Perlin
            niert_loss = self.hparams.o_loss_coff * observed_loss + masked_loss      # Perlin
            self.log("val/val_mae_niert", niert_loss, on_epoch=True)


            return {"val_loss": niert_loss, "val_masked_loss": masked_loss}


        elif self.layout_model == "NIERT_PhysioNet":
            loss = self.criterion(heat_label, heat_pred, masks[0])
            self.log("val/val_mae", loss, on_epoch=True)

            masked_loss = mean_squared_error_inter(heat_label, heat_pred, masks[0], masks[1])
            self.log("val/val_mae_niert", masked_loss, on_epoch=True)

            return {"val_loss": loss, "val_masked_loss": masked_loss}

        else:

            if extra["masks"] is not None:
                loss = self.criterion( heat_label, heat_pred, extra["masks"] ) * self.hparams.std_heat
            else:
                loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat

            self.log("val/val_mae", loss, on_epoch=True)
            return {"val_loss": loss}


    # def validation_epoch_end(self, outputs):
    #     if self.layout_model == "NIERT":
    #         val_loss_mean = torch.stack([x["val_masked_loss"] for x in outputs]).mean()
    #         self.log("val/val_mae_epoch", val_loss_mean.item())

    #         val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
    #         self.log("val/val_mae_niert_epoch", val_loss_mean.item())
    #     else:
    #         val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
    #         self.log("val/val_mae_epoch", val_loss_mean.item())

    def test_step(self, batch, batch_idx):

        if self.layout_model == "NIERT_PhysioNet":
            heat_pred, heat_label, masks = self(batch)
        else:

            masks = None
            if len(batch) == 5:
                obs_index, heat_obs, pred_index, heat, _ = batch
            elif len(batch) == 6:
                obs_index, heat_obs, pred_index, heat, _, masks = batch
            else:
                assert False

            # heat_info = [obs_index, heat_obs, pred_index, heat]
            heat_info = [obs_index, heat_obs, pred_index, heat, masks]
            heat_pred, heat_label, extra = self(heat_info)

            extra["features"] = {k:v for k,v in extra["features"].items() if k in ["o_residual", "t_predict_real"]}

            if masks is not None:
                extra["masks"] = masks

            # if len(batch[0].shape) == 4:
            #     assert batch[0].size(0) == 1
            #     obs_index, heat_obs, pred_index, heat, _ = batch
            #     obs_index, heat_obs, pred_index, heat = obs_index.squeeze(0), heat_obs.squeeze(0), pred_index.squeeze(0), heat.squeeze(0)
            # else:
            #     obs_index, heat_obs, pred_index, heat, _ = batch

            # heat_info = [obs_index, heat_obs, pred_index, heat]
            # heat_pred, heat_label = self(heat_info)


        if self.hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:

            if self.layout_model in ["NIERT", "NIERTPP"]:
                heat_label = heat_label[:, obs_index.size(1):, :]
                heat_pred = heat_pred[:, obs_index.size(1):, :]

            heat_label = heat_label.reshape(-1, 1, 200, 200)
            heat_pred = heat_pred.reshape(-1, 1, 200, 200)

            # heat_label = heat_label.transpose(2, 3)
            # heat_pred = heat_pred.transpose(2, 3)

            # masked_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            # self.log("val/val_mae", masked_loss, on_epoch=True)

            # niert_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            # self.log("val/val_mae_niert", niert_loss, on_epoch=True)

            # return {"val_loss": niert_loss, "val_masked_loss": masked_loss}

            loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat

            default_layout = (
                torch.repeat_interleave(
                    self.default_layout, repeats=heat_pred.size(0), dim=0
                )
                .float()
                .to(device=heat_label.device)
            )
            ones = torch.ones_like(default_layout).to(device=heat_label.device)
            zeros = torch.zeros_like(default_layout).to(device=heat_label.device)
            layout_ind = torch.where(default_layout < 1e-2, zeros, ones)
            loss_2 = (
                torch.sum(torch.abs(torch.sub(heat_label, heat_pred)) * layout_ind)
                * self.hparams.std_heat
                / torch.sum(layout_ind)
            )
            # ---------------------------------
            loss_1 = (
                torch.sum(
                    torch.max(
                        torch.max(
                            torch.max(
                                torch.abs(torch.sub(heat_label, heat_pred)) * layout_ind, 3
                            ).values,
                            2,
                        ).values
                        * self.hparams.std_heat,
                        1,
                    ).values
                )
                / heat_pred.size(0)
            )
            # ---------------------------------
            boundary_ones = torch.zeros_like(default_layout).to(device=heat_label.device)
            boundary_ones[..., -2:, :] = ones[..., -2:, :]
            boundary_ones[..., :2, :] = ones[..., :2, :]
            boundary_ones[..., :, :2] = ones[..., :, :2]
            boundary_ones[..., :, -2:] = ones[..., :, -2:]
            loss_3 = (
                torch.sum(torch.abs(torch.sub(heat_label, heat_pred)) * boundary_ones)
                * self.hparams.std_heat
                / torch.sum(boundary_ones)
            )
            # ----------------------------------

            loss_4 = (
                torch.sum(
                    torch.max(
                        torch.max(
                            torch.max(torch.abs(torch.sub(heat_label, heat_pred)), 3).values, 2
                        ).values
                        * self.hparams.std_heat,
                        1,
                    ).values
                )
                / heat_pred.size(0)
            )

            # pdb.set_trace()

            return {
                "test_loss": loss,
                "test_loss_1": loss_1,
                "test_loss_2": loss_2,
                "test_loss_3": loss_3,
                "test_loss_4": loss_4,
                "task_to_be_saved": (obs_index.cpu(), heat_obs.cpu(), pred_index.cpu(), heat.cpu(), heat_pred.cpu(), extra) if batch_idx < 50 else None
            }

        elif self.hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:
        
            loss = self.criterion(heat_label, heat_pred, masks[0])
            # self.log("val/val_mae", loss, on_epoch=True)

            masked_loss = mean_squared_error_inter(heat_label, heat_pred, masks[0], masks[1])
            # self.log("val/val_mae_niert", masked_loss, on_epoch=True)

            return {
                "val_loss": loss,
                "val_masked_loss": masked_loss,
                "batch_size": heat_label.size(0)
            }

        elif self.hparams.dataset_type in ["Mathit", "Current", "Perlin", "TFR_Scattered"]:

            test_criterion = torch.nn.MSELoss(reduction='none')

            losses_t = test_criterion(heat_label[:, obs_index.size(1):, :], heat_pred[:, obs_index.size(1):, :])
            losses_o = test_criterion(heat_label[:, :obs_index.size(1), :], heat_pred[:, :obs_index.size(1), :])

            losses = test_criterion(heat_label, heat_pred) * self.hparams.std_heat

            losses = losses.squeeze(-1).mean(axis=-1)

            # self.log("val/val_mae", loss, on_epoch=True)

            # RRR = min(heat_pred.size(1), output_0.size(1)) - obs_index.size(1)
            # print(RRR)

            # print(heat_pred[:,obs_index.size(1):obs_index.size(1)+RRR,:].shape, output_0[:,obs_index.size(1):obs_index.size(1)+RRR,:].shape)

            return {
                "val_loss_observed": losses_o,
                "val_loss_target": losses_t,
                "val_loss": losses,
                # "mean_std_rebuttal": torch.abs(heat_pred[:,0:1,:] - output_0[:,0:1,:]).mean(),
                # "mean_std_rebuttal": torch.abs(heat_pred[:,obs_index.size(1):obs_index.size(1)+RRR,:] - output_0[:,obs_index.size(1):obs_index.size(1)+RRR,:]).mean(),
                "given_points": obs_index.size(1),
            }
        
        elif self.hparams.dataset_type in ["PTV"]:

            observed_num = obs_index.size(1)

            # if self.hparams.observed_as_target:
            #     assert False
            #     masked_loss = self.criterion(
            #         heat_label[:, observed_num:-observed_num, :],
            #         heat_pred[:, observed_num:-observed_num, :]
            #     ) * self.hparams.std_heat
            #     self.log("val/val_mae", masked_loss, on_epoch=True)

            #     masked_observed_loss = self.criterion(
            #         heat_label[:, -observed_num:, :],
            #         heat_pred[:, -observed_num:, :]
            #     ) * self.hparams.std_heat
            #     self.log("val/val_observed_as_target_mae", masked_observed_loss, on_epoch=True)

            if self.layout_model in ["NIERT", "NIERTPP"]:
                heat_label = heat_label[:, obs_index.size(1):, :]
                heat_pred = heat_pred[:, obs_index.size(1):, :]

            assert extra["masks"] is not None
            masked_loss = self.criterion(
                heat_label,
                heat_pred,
                extra["masks"]
            ) * self.hparams.std_heat

            # import pdb; pdb.set_trace()

            return {
                "test_loss": masked_loss,
                "task_to_be_saved": (obs_index.cpu(), heat_obs.cpu(), pred_index.cpu(), heat.cpu(), heat_pred.cpu(), extra["masks"].cpu(), extra["features"]) if batch_idx < 50 else None
            }


        else:
            raise NotImplementedError


    def test_epoch_end(self, outputs):

        if self.hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:
            test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
            self.log("test_loss (" + "MAE" + ")", test_loss_mean.item())
            # test_loss_max = torch.max(torch.stack([x["test_loss_1"] for x in outputs]))
            test_loss_max = torch.stack([x["test_loss_1"] for x in outputs]).mean()
            self.log("test_loss_1 (" + "M-CAE" + ")", test_loss_max.item())
            test_loss_com_mean = torch.stack([x["test_loss_2"] for x in outputs]).mean()
            self.log("test_loss_2 (" + "CMAE" + ")", test_loss_com_mean.item())
            test_loss_bc_mean = torch.stack([x["test_loss_3"] for x in outputs]).mean()
            self.log("test_loss_3 (" + "BMAE" + ")", test_loss_bc_mean.item())
            test_loss_max_1 = torch.stack([x["test_loss_4"] for x in outputs]).mean()
            self.log("test_loss_4 (" + "MaxAE" + ")", test_loss_max_1.item())


            # ONLY TFR?
            task_to_be_saved = [x["task_to_be_saved"] for x in outputs if x["task_to_be_saved"]]

            # pdb.set_trace()

            import pickle
            dump_path = "vis_results/%s_HSink_%s_res.pkl" % (self.hparams.dataset_type, self.hparams.model_name)
            with open(dump_path, "wb") as f:
                pickle.dump(task_to_be_saved, f)
            print("Some res dumped to", dump_path)


        elif self.hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:

            test_loss_sum = torch.stack([x["val_masked_loss"] * x["batch_size"] for x in outputs]).sum()
            test_loss_mean = test_loss_sum / sum([x["batch_size"] for x in outputs])
            # test_loss_mean = torch.stack([x["val_masked_loss"] for x in outputs]).mean()
            self.log("MSE Error", test_loss_mean.item())

            return test_loss_mean

        elif self.hparams.dataset_type in ["Mathit", "Current", "Perlin", "TFR_Scattered"]:

            points_nums = set([x["given_points"] for x in outputs])
            loss_by_num = {
                pn:[] for pn in points_nums
            }
            for x in outputs:
                loss_by_num[x["given_points"]].append(x["val_loss"])

            _loss_by_num = {
                pn:torch.cat(loss_by_num[pn]).mean().item() for pn in loss_by_num
            }

            test_loss_mean = torch.stack([x["val_loss"].mean() for x in outputs]).mean()
            test_loss_target_mean = torch.stack([x["val_loss_target"].mean() for x in outputs]).mean()
            test_loss_observed_mean = torch.stack([x["val_loss_observed"].mean() for x in outputs]).mean()

            # test_mean_std_rebuttal = torch.stack([x["mean_std_rebuttal"].mean() for x in outputs]).mean()

            print("losses_by_num = ", _loss_by_num)
            print("val_loss = ", test_loss_mean.item())
            print("val_loss_target = ", test_loss_target_mean.item())
            print("val_loss_observed = ", test_loss_observed_mean.item())


            # print("Mean std:", test_mean_std_rebuttal.item())
            # print("Mean std:", test_mean_std_rebuttal.item())
            # print("Mean std:", test_mean_std_rebuttal.item())
            # print("Mean std:", test_mean_std_rebuttal.item())

            self.log("val_loss = ", test_loss_mean.item())


            # task_to_be_saved = [x["task_to_be_saved"] for x in outputs if x["task_to_be_saved"]]

        elif self.hparams.dataset_type in ["PTV"]:
            test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
            self.log("test_loss (" + "MSE" + ")", test_loss_mean.item())
            # test_loss_max = torch.max(torch.stack([x["test_loss_1"] for x in outputs]))

            # ONLY TFR?
            task_to_be_saved = [x["task_to_be_saved"] for x in outputs if x["task_to_be_saved"]]

            import pickle
            dump_path = "vis_results/%s_%s_NEW_res.pkl" % (self.hparams.dataset_type, self.hparams.model_name)
            with open(dump_path, "wb") as f:
                pickle.dump(task_to_be_saved, f)
            print("Some res dumped to", dump_path)


        else:
            raise NotImplementedError




    @staticmethod
    def add_model_specific_args(parser):  # pragma: no-cover
        """Parameters you define here will be available to your model through `self.hparams`."""
        # dataset args
        parser.add_argument(
            "--data_root", type=str, required=True, help="path of dataset"
        )
        parser.add_argument(
            "--train_path", type=str, required=True, help="path of train dataset list"
        )
        parser.add_argument(
            "--train_size",
            default=0.8,
            type=float,
            help="train_size in train_test_split",
        )
        parser.add_argument(
            "--test_path", type=str, required=True, help="path of test dataset list"
        )
        # parser.add_argument("--boundary", type=str, default="rm_wall", help="boundary condition")
        parser.add_argument(
            "--data_format",
            type=str,
            default="mat",
            choices=["mat"],
            help="dataset format",
        )

        # Normalization params
        parser.add_argument("--mean_heat", default=0, type=float)
        parser.add_argument("--std_heat", default=1, type=float)

        # Model params (opt)
        parser.add_argument(
            "--model_name", type=str, default="FCN", help="the name of chosen model"
        )

        return parser
