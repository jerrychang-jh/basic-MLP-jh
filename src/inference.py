import logging
import numpy as np
import torch
from torch import nn

from models.net import MLP
import config

def eval(dataloader, model, loss_fn):
    model.eval()
    n_batch = len(dataloader)
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    total_loss /= n_batch
    correct /= n_batch

    print(f"Eval: Accuracy: {(correct * 100):>0.1f}% Avg loss: {total_loss:>8f}")

if __name__ == '__main__':

    eval_dataloader = np.random.rand((32, 10, 10))

    model = MLP()
    loss_fn = nn.MSELoss()

    for epoch in range(config.epochs):
        eval(eval_dataloader, model)
        
    print("Done")
        