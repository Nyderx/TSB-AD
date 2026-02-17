import logging
import traceback
from typing import Dict, Any

import torchinfo
import tqdm, math
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import pytorch_lightning as pl

from models.xlstmad_tirex.xLSTM import xLSTM
from utils.dataset import ReconstructDataset


class xLSTMADTirexModule(pl.LightningModule):
    def __init__(
        self,
        # window_size,
        feats,
        hidden_dim,
        pred_len,
        num_layers,
        # batch_size,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.feats = feats
        self.lr = lr
        self.num_layers = num_layers

        self.xlstm_encoders = nn.ModuleList(
            [
                xLSTM(
                    input_size=feats if i == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    seq_len=pred_len,
                )
                for i in range(num_layers)
            ]
        )

        self.xlstm_decoders = nn.ModuleList(
            [
                xLSTM(
                    input_size=feats if i == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    seq_len=pred_len,
                )
                for i in range(num_layers)
            ]
        )

        self.relu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, feats)

        self.train_loss = nn.MSELoss()
        self.val_loss = nn.MSELoss()
        self.save_hyperparameters()


    def forward(self, x):
        try:
            cur_batch_size = x.shape[0]
            cur_seq_len = x.shape[1]

            xlstm_encoder_states = [None] * self.num_layers
            for i in range(self.num_layers):
                x, xlstm_encoder_states[i] = self.xlstm_encoders[i](x)

            xlstm_decoder_input = torch.zeros(cur_batch_size, cur_seq_len, self.feats).to(
                self.device
            )

            for i in range(self.num_layers):
                xlstm_decoder_input, xlstm_encoder_states[i] = self.xlstm_decoders[i](
                    xlstm_decoder_input, xlstm_encoder_states[i]
                )

            xlstm_decoder_input = self.relu(xlstm_decoder_input)
            xlstm_outputs = self.fc(xlstm_decoder_input)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise e

        return xlstm_outputs

    def training_step(self, batch, batch_idx):
        x, weights = batch
        x_hat = self(x)
        loss = self.train_loss(x_hat, x)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.val_loss(x_hat, x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_test_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        self.test_outputs.append(((x - x_hat) ** 2).cpu().numpy().mean(axis=2))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx) -> Any:
        x, target = batch
        reconstruction = self.forward(x)
        anomaly_scores = torch.mean(mse_loss(reconstruction, target, reduction='none'), dim=(1, 2))
        return anomaly_scores


class xLSTMADTirex:
    def __init__(self, model: pl.LightningModule, window_size: int = 100, validation_size: float = 0.2,
                 batch_size: int = 128):
        self.window_size = window_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.model = model

    def fit(self, data):
        train_data = data[:int((1 - self.validation_size) * len(data))]
        valid_data = data[int((1 - self.validation_size) * len(data)):]


        print('train data size:', len(train_data))
        print('valid data size:', len(valid_data))

        train_loader = DataLoader(
            ReconstructDataset(train_data, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        valid_loader = DataLoader(
            ReconstructDataset(valid_data, window_size=self.window_size),
            batch_size=4 * self.batch_size,
            shuffle=False,
            num_workers=4
        )

        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min")

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=1e-4),
                checkpoint_cb],
            logger=True,
            enable_progress_bar=True,
            limit_train_batches=5
        )

        print(f'Trainer log file {trainer.log_dir}')
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        print(f'Loading best model from {checkpoint_cb.best_model_path}')
        self.model = self.model.__class__.load_from_checkpoint(checkpoint_cb.best_model_path)

    def decision_function(self, data):
        data_loader = DataLoader(
            ReconstructDataset(data, window_size=self.window_size),
            batch_size=4 * self.batch_size,
            shuffle=False,
            num_workers=4,
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            logger=True,
            enable_checkpointing=False,
            # limit_predict_batches=3
        )

        self.model.eval()
        anomaly_scores = np.zeros(len(data))

        with torch.no_grad():
            preds = trainer.predict(self.model, dataloaders=data_loader)

        scores = torch.concat(preds)
        if scores.shape[0] < len(data):
            logging.info("Adjusting anomaly scores length to match data length.")
            padded_decision_scores = np.zeros(len(data))
            padded_decision_scores[: self.window_size - 1] = scores[0]
            padded_decision_scores[self.window_size- 1:] = scores
            return padded_decision_scores

        return scores.numpy()

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))