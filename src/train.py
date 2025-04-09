import logging
import numpy as np
import torch
from torch import nn

from models.net import MLP
import config

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>3f}")

def test(dataloader, model, loss_fn):
    model.eval()

if __name__ == '__main__':

    train_dataloader = np.random.rand((32, 10, 10))

    model = MLP()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        
    print("Done")
        