from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import optim, FloatTensor

from .model.loa import LOANet
from .utils import FocalLoss
from .utils import StreamSegMetrics


class LitLOA(pl.LightningModule):
    def __init__(
            self,
            alpha: float,
            gamma: int,
            size_average: bool,
            num_classes: int,
            in_channels: int,
            growth_rate: int,
            num_layers: Tuple[int, int, int, int],
            reduction: float,
            p_channels: int,
            u_channels: int,
            dropout_rate: float,
            attention_mid_channels: int,
            attention_scale: int,
            learning_rate: float,
            patience: int
    ):
        super().__init__()
        self.save_hyperparameters()
        self.focalLoss = FocalLoss(alpha=alpha,
                                   gamma=gamma,
                                   size_average=size_average)

        self.loaNet = LOANet(num_classes=num_classes,
                             in_channels=in_channels,
                             growth_rate=growth_rate,
                             num_layers=num_layers,
                             reduction=reduction,
                             p_channels=p_channels,
                             u_channels=u_channels,
                             dropout_rate=dropout_rate,
                             attention_mid_channels=attention_mid_channels,
                             attention_scale=attention_scale)

        self.train_metrics = StreamSegMetrics(num_classes)
        self.val_metrics = StreamSegMetrics(num_classes)
        self.test_metrics = StreamSegMetrics(num_classes)

    def forward(self, image) -> FloatTensor:
        return self.loaNet(image)

    # def on_train_start(self) -> None:
    #     ckpt = torch.load("")
    #     state_dict = ckpt["state_dict"]
    #     new_state_dict = {}
    #     for s_key in state_dict:
    #         # if s_key[:16] == "ledcNet.backbone":
    #         new_state_dict[s_key[8:]] = state_dict[s_key]
    #     self.ledcNet.load_state_dict(new_state_dict, strict=False)

    def on_train_epoch_start(self) -> None:
        self.train_metrics.reset()

    def training_step(self, batch, _):
        images = batch[0]
        labels = batch[1]

        out, out_aux = self(images)
        loss = self.focalLoss(out_aux, labels) + self.focalLoss(out, labels)

        preds1 = out.clone().detach().max(dim=1)[1].cpu().numpy()
        labels1 = labels.clone().cpu().numpy()

        self.train_metrics.update(labels1, preds1)
        self.log("loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        score = self.train_metrics.get_results()
        self._save_log(score, "t")

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()

    def validation_step(self, batch: list, _):
        image = batch[0]
        label = batch[1]

        outputs = self(image)
        pred = outputs.clone().detach().max(dim=1)[1].cpu().numpy()
        label = label.clone().cpu().numpy()
        self.val_metrics.update(label, pred)

    def on_validation_epoch_end(self) -> None:
        score = self.val_metrics.get_results()
        self._save_log(score, "v")

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()

    def test_step(self, batch: list, _):
        image = batch[0]
        label = batch[1]

        outputs = self(image)
        pred = outputs.clone().detach().max(dim=1)[1].cpu().numpy()
        label = label.clone().cpu().numpy()
        self.test_metrics.update(label, pred)

    def on_test_epoch_end(self) -> None:
        score = self.test_metrics.get_results()
        self._save_log(score, "tst")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-4,
            weight_decay=1e-4
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "v_acc",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _save_log(self, score, run_type):

        for metric_type in score:
            name = run_type + "_" + metric_type
            prog_bar = False
            on_step = False
            on_epoch = True
            sync_dist = True
            if run_type != "tst":
                if metric_type == "m_IoU":
                    prog_bar = True

            if type(score[metric_type]) == dict:
                for metric_type_class in score[metric_type]:
                    name_cls = name + "_" + str(metric_type_class)
                    self.log(name_cls, score[metric_type][metric_type_class],
                             prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
            else:
                self.log(name, score[metric_type],
                         prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
