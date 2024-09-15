import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

num_records = 100000
num_features = 16
num_classes = 4

dataX = np.random.randn(num_records, num_features)
dataY = np.concatenate([[c] * (num_records // num_classes) for c in range(num_classes)])
X_train, X_test, y_train, y_test = train_test_split(
    dataX, 
    dataY, 
    test_size=0.3,  
    random_state=42,  
    stratify=dataY
)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return {
            'X': torch.tensor(self.X[index], dtype=torch.float32),
            'y': torch.tensor(self.y[index], dtype=torch.int16)
        }
    
train_dataset = SimpleDataset(X_train, y_train)
test_dataset = SimpleDataset(X_test, y_test)
         
        


        