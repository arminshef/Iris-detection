import torch
from sklearn.preprocessing import LabelEncoder

class Dataset(torch.utils.data.Dataset):
    def __init__(self, Data):
        self.x = Data.values[:,:-1]
        self.y = Data.values[:,-1]
        self.x = self.x.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return [self.x[idx], self.y[idx]]
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        # calculate the split
        return torch.utils.data.random_split(self, [train_size, test_size])