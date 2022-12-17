from utils.losses import triplet_semihard_loss
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch import Tensor
from utils.dataset import IndividualsDS
from sklearn.model_selection import train_test_split#
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear, Sequential, AdaptiveAvgPool2d
import torch
from torchvision.models import inception_v3, Inception_V3_Weights

class TripletLoss(pl.LightningModule):
    def __init__(self, df, embedding_size, batch_size=32, lr=0.00001, ):
        super(TripletLoss, self).__init__()
        self.df = df
        self.batch_size = batch_size
        self.lr = lr
        
        self.trainAcc = Accuracy()
        self.valAcc = Accuracy()

        #backend
        # ToDo use pretrained weights
        self.backend = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        
        # "frontend"
        self.frontend  = Sequential(
            AdaptiveAvgPool2d((1,1)), # not sure if this is exactly the size we need
            Linear(self.backend.out_channels, embedding_size) #in = size of out of backend| out = embedding size
        )
    
    def forward(self, x: Tensor):
        x = self.backend(x)
        x = self.frontend(x)
        return x

    def prepare_data(self):
        df = self.df
        train, validate = train_test_split(df, test_size=0.3, random_state=0, stratify=df['label'])
        self.trainDF = IndividualsDS(train)
        self.validateDF = IndividualsDS(validate)
        # TODO: might need to do some cross validation?

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.0)

    def train_dataloader(self):
        # ToDo: Implement data augmentation as in the paper
        return DataLoader(self.trainDF, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validateDF, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch: dict, _batch_idx: int):
        inputs, labels = batch['image'], batch['labels']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = triplet_semihard_loss(outputs, labels)
        self.trainAcc(outputs.argmax(dim=1), labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_epoch_end(self):
        self.log('train_acc', self.trainAcc.compute() * 100, prog_bar=True)
        self.trainAcc.reset()

    def validation_step(self, batch: dict, _batch_idx: int):
        inputs, labels = batch['image'], batch['labels']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = triplet_semihard_loss(outputs, labels)
        
        self.valAcc(outputs.argmax(dim=1), labels)
        
        return {'val_loss': loss}
    
    def validation_epoch_end(self, validationStepOutputs):
        avgLoss = torch.stack([x['val_loss'] for x in validationStepOutputs]).mean()
        valAcc = self.valAcc.compute() * 100
        self.valAcc.reset()
        self.log('val_loss', avgLoss, prog_bar=True)
        self.log('val_acc', valAcc, prog_bar=True)