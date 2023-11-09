import torch
import torchaudio
import numpy as np
import soundfile as sf
from torch.nn import functional as torchf
from torch import nn
from VocalSetDataset import VocalSetDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt



# TODO add trim transform

class PadTransform:
    def __init__(self, pad_to):
        self.pad_to = pad_to
        self.transform = torchf.pad

    def __call__(self, tensor):
        amount_to_pad = self.pad_to - tensor.shape[0]
        return self.transform(tensor, amount_to_pad)


class OneHotTransform:
    def __init__(self, num_classes):
        self.transform = torchf.one_hot
        self.num_classes = num_classes

    def __call__(self, label):
        tensor = torch.tensor(label)  # TODO add error checking etc
        return self.transform(tensor, num_classes=self.num_classes)


class Average(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, X):
        avg = torch.mean(X, dim=self.dim, keepdim=self.keep_dim)
        return avg


# DEBUG
class Printer(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,X):
    print(X.shape)
    return X

class FullConv(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 80, 4),  # size
            nn.MaxPool1d(4, 4),
            #Printer(),
            nn.BatchNorm1d(64, momentum=None),  # TODO FIGURE OUT THESE ARGS
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.Conv1d(64, 64, 3, 1),
            nn.MaxPool1d(4, 4),
            #Printer(),
            nn.BatchNorm1d(64, momentum=None),  # TODO FIGURE OUT THESE ARGS
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 1),
            nn.Conv1d(128, 128, 3, 1),
            nn.MaxPool1d(4, 4),
            #Printer(),
            nn.BatchNorm1d(128, momentum=None),  # TODO FIGURE OUT THESE ARGS
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, 1),
            nn.Conv1d(256, 256, 3, 1),
            nn.Conv1d(256, 256, 3, 1),
            nn.MaxPool1d(4, 4),
            #Printer(),
            nn.BatchNorm1d(256, momentum=None),  # TODO FIGURE OUT THESE ARGS
            nn.Conv1d(256, 512, 3, 1),
            nn.Conv1d(512, 512, 3, 1),
            Average(1, keep_dim=True),
            Printer(),
            nn.Linear(822, out_classes)
        )

        def forward(self, X):
            out = self.net(X)
            return out


if __name__ == "__main__":

    # TODO PREPROCESS DATA BETTER :(
    DATA_LABEL_PATH = r"D:\pjmcc\Documents\OSU_datasets\VocalSet"
    MAX_DURATION = 848104
    num_classes = 5
    feature_transform = PadTransform(MAX_DURATION)
    label_transform = OneHotTransform(num_classes=num_classes)

    dataset = VocalSetDataset("VocalSet/annotations.txt",transform=feature_transform, target_transform=label_transform)
    train_set, val_set = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=16, shuffle=True)

    out_classes = 5
    model = FullConv(out_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    loss_list = []
    acc_list = []
    N = len(train_dataloader)
    lossf = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.to(device)
    for i in range(epochs):
        train_loss = 0
        acc = 0
        for X, y in tqdm(iter(train_dataloader)):
            X = X.to(device)
            y = y.to(device)

            preds = model(X)

            loss = lossf(preds, y)
            train_loss += loss.item()
            acc += (np.argmax(preds) == np.argmax(y)).sum()

            loss.backward()

            optim.step()
            optim.zero_grad()

        avg_train_loss = train_loss / N
        epoch_acc = acc / N
        print(epoch_acc)
        acc_list.append(epoch_acc)
        loss_list.append(avg_train_loss)


    plt.plot(np.arange(epochs), loss_list)
    plt.show()