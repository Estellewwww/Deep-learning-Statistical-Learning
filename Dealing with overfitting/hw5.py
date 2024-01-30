import sklearn.datasets
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def make_dataset(version=None, test=False):
    if test:
        random_state = None
    else:
        random_states = [27,33,38]
        if version is None:
            version = random.choice(range(len(random_states)))
            print(f"Dataset number: {version}")
        random_state = random_states[version]
    return sklearn.datasets.make_circles(factor=0.7, noise=0.1, random_state=random_state)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 500) 
        self.fc2 = nn.Linear(500, 500) 
        self.fc3 = nn.Linear(500, 2) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def convert_data(X_train,y_train,X_test,y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset
    
def train_model(model, train_loader, epochs=1000):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
    return target, predicted

def manual_clas(point, center, threshold):
    distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    return 0 if distance > threshold else 1
    
    
if __name__ == '__main__':
    #a
    X_train, y_train = make_dataset(version=0, test=False)
    X_test, y_test = make_dataset(test=True)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train,test = convert_data(X_train,y_train,X_test,y_test)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)
    
    net = Net()
    
    train_model(net, train_loader)
    
    y_true,y_pred = evaluate_model(net, test_loader)

    print("The accuracy is: ",accuracy_score(y_true, y_pred))
    
    #b
    plt.figure(figsize=(8, 6))
    # Correctly classified points
    plt.plot(y_true, marker='o', color='green', label='Acutual')
    # Incorrectly classified points
    plt.plot(y_pred, marker='x', color='red', label='Prediction')
    plt.legend()
    plt.show()
    
    #c
    
    x1_min, x1_max = X_train[:, 0].min()-0.5 , X_train[:, 0].max()+0.5 
    x2_min, x2_max = X_train[:, 1].min()-0.5 , X_train[:, 1].max()+0.5
    h = 0.01

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    net.eval()
    with torch.no_grad():
        Z = net(torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32))
        Z = Z.max(1)[1].reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Spectral)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #d
    center = [0, 0]
    threshold = np.sqrt(1.65)
    
    pred = []
    for i in range(len(X_test)):
        point = X_test[i]
        pred.append(manual_clas(point, center, threshold))
    
    print(accuracy_score(y_test,pred))
    
    
    
    
    
    
    