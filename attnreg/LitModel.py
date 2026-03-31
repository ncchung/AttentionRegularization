# from dino_trunc import dino_trunc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class LitModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        # self.model = dino_trunc()
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        # only train linear layer
        for p in self.model.parameters():
            p.requires_grad = False
        self.linear = nn.Linear(384, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    # def forward(self, x):
    #     x = self.model(x)
    #     x = self.linear(x)
    #     return x

    def prepare_tokens(self, x, remove=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
    
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
    
        # remove patch embeddings defined in "remove", this is the new part
        num_patches = x.shape[1]
        if remove != None:
            t = torch.arange(num_patches)
            idx_to_keep = [i for j, i in enumerate(t) if j not in remove]
            x = x[:, idx_to_keep, :]
        return x

    def forward(self, x, remove=None):
        x = self.prepare_tokens(x, remove)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x[:, 0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

   