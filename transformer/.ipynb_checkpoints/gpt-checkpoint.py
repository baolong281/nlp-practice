import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

#hyperparameters
context_window = 64
train_iters = 7000
learning_rate = .0003
head_size = 16


with open('text.txt', 'r', encoding=('utf-8')) as f:
    text = f.read()

chars = sorted(list(set(text)))
n_embd = len(chars)


stoi = {ch:i for i, ch in enumerate(chars)}
itoc = {i:ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itoc[c] for c in x])

data = torch.tensor(encode(text))

#train test
train_size = int(len(data) * .9)
test_size = len(data) - train_size
batch_size = 32

train, test = random_split(data, [train_size, test_size])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -.05
        return wei

model = Head(n_embd)

for x in train_loader:
    print(x.shape)
    y = model(x)
    print(y)
    break



