from abc import abstractmethod, abstractstaticmethod
from typing import Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

def binary_accuracy(ypred, y):
    return sum(ypred.round() == y)/float(y.shape[0])


def sklearn_logreg(X_train, y_train, X_test, y_test):
    sk_logr = LogisticRegression(fit_intercept=False, penalty='none')
    sk_logr.fit(X_train, y_train)
    return binary_accuracy(sk_logr.predict(X_test), y_test)


class HW1Data():
    @abstractmethod
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def data_split(self, test_size=0.33) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class SkLearnGenerator(HW1Data):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    @abstractstaticmethod
    def _generator(n_samples) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def data(self):
        return type(self)._generator(self.n_samples)

    def data_split(self, test_size=0.33):
        X, y = self.data()
        return train_test_split(X, y, test_size=test_size)


class Make_classification(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_classification(n_samples, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)


class Make_moons(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_moons(n_samples, noise=0.05)


class Make_circles(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_circles(n_samples, factor=0.5, noise=0.05)
    
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),  
            nn.ReLU(),                 
            nn.Linear(64, 32),         
            nn.ReLU(),                 
            nn.Linear(32, 1),          
            nn.Sigmoid()               
        )
    
    def forward(self, x):
        return self.network(x)
    
def pytorch_loss(data_gen,model):
    X_train, X_test, y_train, y_test = data_gen.data_split()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_data = TensorDataset(X_train_t, y_train_t.view(-1, 1))
    train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Zero gradients, backward pass, and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print loss every 10 epochs
        #if epoch % 10 == 0:
            #print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')
    
    # Evaluate the model
    with torch.no_grad():
        y_pred_train = model(X_train_t)
        train_loss = criterion(y_pred_train, y_train_t.view(-1, 1))
        y_pred_test = model(X_test_t)
        test_loss = criterion(y_pred_test, y_test_t.view(-1, 1))
        print(f'Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')
        
    return test_loss
        
def sk_loss(data_gen):
    X_train, X_test, y_train, y_test = data_gen.data_split()
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]
    
    train_loss = log_loss(y_train, pred_train)
    test_loss = log_loss(y_test, pred_test)
    
    print(f'Scikit-learn Train Log Loss: {train_loss}')
    print(f'Scikit-learn Test Log Loss: {test_loss}')
    
    return test_loss
    

    
if __name__ == '__main__':
    
    #a
    #pytorch
    n_samples = 100  # You can set the number of samples here
    MClass = Make_classification(n_samples)
    model = LogisticRegressionModel()
    
    print("a.")
    print("Result for Pytorch:")
    loss_class_p = pytorch_loss(MClass,model)
    
        
    print()
    print("Result for Sklearn:")
    loss_class_s = sk_loss(MClass)
    
    
    #b
    #make moons
    Mmoon = Make_moons(n_samples)
    
    print()
    print("b.")
    print("Make_moons:")
    print()
    print("Result for Pytorch:")
    loss_moon_p = pytorch_loss(Mmoon,model)
    
        
    print()
    print("Result for Sklearn:")
    loss_moon_s = sk_loss(Mmoon)
    
    #make_circle
    Mcircle = Make_circles(n_samples)
    
    print()
    print("Make_circle:")
    print()
    print("Result for Pytorch:")
    loss_circle_p = pytorch_loss(Mcircle,model)
    
        
    print()
    print("Result for Sklearn:")
    loss_circle_s = sk_loss(Mcircle)
    
    loss_p = [loss_class_p,loss_moon_p,loss_circle_p]
    loss_s = [loss_class_s,loss_moon_s,loss_moon_s]
    loss = {"pytorch":loss_p, "sklearn":loss_s}
    loss = pd.DataFrame(loss)
    
    plt.plot(loss)
    plt.legend()
    plt.show()
    
    #The loss increase when change the dataset
    
    #c
    model_deep = DeepNN()
    
    print()
    print("c.")
    print("Result for classification data:")
    loss_class_deep = pytorch_loss(MClass,model_deep)
    
    print()
    print("Result for Make moon date:")
    loss_moon_deep = pytorch_loss(Mmoon,model_deep)
    
    print()
    print("Result for Make circle data:")
    loss_circle_deep = pytorch_loss(Mcircle,model_deep)
    
    loss["deep"] = [loss_class_deep,loss_moon_deep,loss_circle_deep]
    plt.plot(loss)
    plt.legend()
    plt.show()
    
    #The loss decrease
    
    
    

