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
import pandas as pd
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
from typing import Tuple
import numpy as np
from .inception import inception_modified as test
from .inception import InceptionOutputs

class TripletLoss(pl.LightningModule):
    def __init__(self, df:pd.DataFrame, embedding_size, img_size: Tuple[int, int]=[300,300], batch_size=32, lr=0.00001, ):
        super(TripletLoss, self).__init__()
        self.df = df
        self.batch_size = batch_size
        self.lr = lr
        self.img_size = img_size
        num_classes=self.df["labels_numeric"].nunique()
        self.valAcc = Accuracy("multiclass", num_classes=num_classes)
        self.trainAcc = Accuracy("multiclass", num_classes=num_classes)

        #backend
        # ToDo use pretrained weights
        self.backend = test(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.backend.eval()
        # "frontend"
        #print(self.backend.fc.out_features)
        self.pooling = AdaptiveAvgPool2d((5,5))
        self.linear = Linear(2048*5*5, embedding_size)
        #self.frontend  = Sequential(
        #    AdaptiveAvgPool2d((5,5)) # not sure if this is exactly the size we need
        #    #Linear(self.backend.fc.out_features, embedding_size) #in = size of out of backend| out = embedding size
        #)
    
    def forward(self, x: Tensor):
        preprocess =    transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        #x = preprocess(x)
        #x = x.unsqueeze(0)
        x = self.backend(x)
        if isinstance(x, InceptionOutputs):
            x = x.logits
        #x = x.view(self.batch_size, 1000, 1, 1)
        #x = x.unsqueeze(2)

        x = self.pooling(x)
        #print("shape0",x.shape)
        x = x.flatten(start_dim=1)
        #print("shape1",x.shape)
        x = self.linear(x)
        #print("shape2",x.shape)
        #print("DGHJWGDHJWG")
        return x

    def prepare_data(self):
        train, validate = train_test_split(self.df, test_size=0.3, random_state=0, stratify=self.df['labels_numeric'])
        self.trainDF = IndividualsDS(train, self.img_size)
        self.validateDF = IndividualsDS(validate, self.img_size)
        # TODO: might need to do some cross validation?

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.0)

    def train_dataloader(self):
        # ToDo: Implement data augmentation as in the paper
        return DataLoader(self.trainDF, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validateDF, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch: dict, _batch_idx: int):
        inputs, labels = batch['images'], batch['labels']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = triplet_semihard_loss(labels, outputs, 'cuda:0')
        print("loss", loss)
        self.trainAcc(outputs.argmax(dim=1), labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_epoch_end(self):
        self.log('train_acc', self.trainAcc.compute() * 100, prog_bar=True)
        self.trainAcc.reset()

    def validation_step(self, batch: dict, _batch_idx: int):
        inputs, labels = batch['images'], batch['labels']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = triplet_semihard_loss(labels, outputs, 'cuda:0')
        
        self.valAcc(outputs.argmax(dim=1), labels)
        
        return {'val_loss': loss}
    
    def validation_epoch_end(self, validationStepOutputs):
        avgLoss = torch.stack([x['val_loss'] for x in validationStepOutputs]).mean()
        valAcc = self.valAcc.compute() * 100
        self.valAcc.reset()
        self.log('val_loss', avgLoss, prog_bar=True)
        self.log('val_acc', valAcc, prog_bar=True)
