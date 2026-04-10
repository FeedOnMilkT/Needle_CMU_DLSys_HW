import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
from needle.autograd import Tensor
np.random.seed(0)

def ConvBN(in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
    return nn.Sequential(
        nn.Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, device=device, dtype=dtype),
        nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
        nn.ReLU(),
    )

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)

        self.res1 = nn.Sequential(
            ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
            ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
        )

        self.conv3 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.conv4 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)

        self.res2 = nn.Sequential(
            ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
            ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
        )

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.res1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = x + self.res2(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.embedding_size = embedding_size

        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)

        if seq_model == 'rnn':
            self.sequence_model = nn.RNN(
                embedding_size, hidden_size, num_layers=num_layers,
                device=device, dtype=dtype
            )
        elif seq_model == 'lstm':
            self.sequence_model = nn.LSTM(
                embedding_size, hidden_size, num_layers=num_layers,
                device=device, dtype=dtype
            )
        else:
            raise ValueError("seq_model must be 'rnn' or 'lstm'")

        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        emb = self.embedding(x)                          # (seq_len, bs, embed_dim)
        out, h = self.sequence_model(emb, h)            # (seq_len, bs, hidden_size)

        seq_len, bs, hidden = out.shape
        out = out.reshape((seq_len * bs, hidden))
        logits = self.linear(out)                       # (seq_len * bs, output_size)

        return logits, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
