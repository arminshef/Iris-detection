import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Read_Dataset
import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from torch.nn.init import xavier_uniform_

t = []
with open('../IRIS/iris.txt') as txt:
    r = txt.readlines()
    for row in r:
        row = row.split('\n')[0]
        p = row.split(',')
        t.append(p)

df = pd.DataFrame(t, columns=['Septal-Length','Septal-Width','petal-Length','Petal-Width','Species'])

def prepare_data(data_file):
    dataset = Read_Dataset(data_file)
    train, test = dataset.get_splits()
    train_dl = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True)
    test_dl  = torch.utils.data.DataLoader(test, batch_size = 1024, shuffle = False)
    return train_dl, test_dl    

def train_model(train_dl, model):
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    for epoch in range(500):
        for i , (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            y_h = model(inputs)
            loss = loss_func(y_h, targets)
            loss.backward()
            optimizer.step()

def eval_model(test_dl, model):
    predictions, actuals = list(), list()
    for i , (inputs, targets) in enumerate(test_dl):
        y_h = model(inputs)
        y_h = y_h.detach().numpy()
        actual = targets.numpy()
        y_h = np.argmax(y_h, axis=1)
        actual = actual.reshape((len(actual), 1))
        y_h = y_h.reshape((len(y_h), 1))
        predictions.append(y_h)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    acc =  accuracy_score(actuals, predictions) *100
    return acc

def predict(row, model):
    row = torch.Tensor([row])
    y_h = model(row)
    y_h = y_h.detach().numpy()
    return y_h

path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = Model(1)
# # train the model
train_model(train_dl, model)
# evaluate the model
acc = eval_model(test_dl, model)
print('Accuracy: %.3f' % acc)