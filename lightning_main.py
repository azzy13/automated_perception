import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import importlib
import lightning as L

# define the LightningModule
class Tracker(L.LightningModule):
    def __init__(self, model_selection, model_params, losses, loss_training, hyperparams):
        super().__init__()
        self.model_basis = importlib.import_module(f"models.{model_selection}")
        self.model = self.model_basis.Model(model_params)
        self.loss_bases = {}
        self.loss_list = {}
        for loss in losses:
            self.loss_bases[loss] = importlib.import_module(f"losses.{loss}")
            self.loss_list[loss] = self.loss_bases[loss].Loss(losses[loss])
        self.loss_training = loss_training
        self.lr = hyperparams["lr"]

    def compute_losses(self, y, y_hat):
        computed_losses = {}
        for loss in self.loss_list:
            computed_loss = self.loss_list[loss](y, y_hat)
            computed_losses[loss] = computed_loss
            self.log(loss, computed_loss)
        return computed_losses[self.loss_training]
    
    def forward(self, x, labels):
        return self.model(x, labels)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, info, id = batch
        y = y[:, :, :5]
        y_hat = self(x, y)
        loss = self.compute_losses(y, y_hat)
        return loss
    
    '''
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, info, id = batch
        print(x.shape, y.shape)
        y = y[:, :, :5]
        print(x.shape, y.shape)
        y_hat = self(x, y)
        loss = self.compute_losses(y, y_hat)
        return loss
    
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, info, id = batch
        y = y[:, :, :5]
        y_hat = self(x, y)
        loss = self.compute_losses(y, y_hat)
        return loss
    '''

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
