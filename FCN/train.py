import os
import pandas as pd
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from Utils import data, train
import numpy as np
from torch import nn
from Utils.model import bilinear_kernel
from Utils.train import loss, updater, train_batch
from model import get_FCN

writer = SummaryWriter()

# Load data.
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = data.load_data_voc(batch_size, crop_size)
imgs, targets = next(iter(train_iter))

# Model.
num_classes = 21
net = get_FCN(num_classes=num_classes)
writer.add_graph(net, imgs)

# Train.
num_epochs, lr, wd, devices = 30, 0.001, 1e-3, train.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
timer, num_batches = train.Timer(), len(train_iter)

net = nn.DataParallel(net, device_ids=devices).to(devices[0])
for epoch in range(num_epochs):
    # Sum of training loss, sum of training accuracy, no. of examples,
    # no. of predictions
    metric = train.Accumulator(4)
    for i, (features, labels) in enumerate(train_iter):
        timer.start()
        l, acc = train_batch(net, features, labels, loss, trainer, devices)
        metric.add(l, acc, labels.shape[0], labels.numel())
        timer.stop()

    train_loss = metric[0] / metric[2]
    train_acc = metric[1] / metric[3]
    test_acc = train.evaluate_accuracy_gpu(net, test_iter)
    writer.add_scalar('Train Loss', train_loss, epoch)
    writer.add_scalar('Train Accuracy', train_acc, epoch)
    writer.add_scalar('Test Accuracy', test_acc, epoch)
    torch.save(net.state_dict(), 'checkpoints/fcn{}.pt'.format(epoch))

print(f'loss {metric[0] / metric[2]:.3f}, train acc '
      f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
      f'{str(devices)}')

writer.close()
