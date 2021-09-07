import torch
from torch.nn.init import xavier_uniform_

class Model(torch.nn.Module):
    def __init__(self, num_inp):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inp, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(10, 8)
        xavier_uniform_(self.hidden1.weight)
        self.act2 = torch.nn.ReLU()
        self.hidden3 = torch.nn.Linear(8,3)
        xavier_uniform_(self.hidden1.weight)
        self.act3 = torch.nn.Softmax(dim=1)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        return x