#%%
import pandas as pd
from tqdm import tqdm
from datetime import date
from dotenv import load_dotenv
import os
import textwrap

import openai
from matplotlib import pyplot as plt
import plotly.express as px
import pickle

import numpy as np
import transformers

import torch
import torch.nn as nn
import torch.optim as optim

completion_labels = pd.read_csv("250completionlabels.csv")

labels250 = completion_labels["label"].values

#%%
embeds = pickle.load(open("../databases/all_pubmed_compl_embeds.p", "rb"))

embeds_250 = [e[1] for e in embeds[250:500]]
#%%

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Set hyperparameters
input_dim = 1536
hidden_dim = 128
output_dim = 1
lr = 0.001
num_epochs = 500

# Instantiate the model and the optimizer
model = MLP(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define the loss function
criterion = nn.BCELoss()

# Define the training loop
def train(model, optimizer, criterion, train_loader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss))
        
# Load the data
# Here, X is a list of 1536-dimensional embeddings, and Y is a list of binary labels
N = 200
X = embeds_250[:N]
Y = labels250[:N]
# Code to load data into X and Y

# Convert the data into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

# Combine X and Y into a TensorDataset
dataset = torch.utils.data.TensorDataset(X, Y)

# Create a DataLoader to handle minibatches during training
batch_size = 50
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
train(model, optimizer, criterion, train_loader, num_epochs)

#%%
# Evaluate the model on the train set
outs = []
with torch.no_grad():
    for x, y in zip(embeds_250[N:], labels250[N:]):
        o = model(torch.tensor(x, dtype = torch.float32))
        
        outs.append(torch.round(o).item())
#%%
sum(outs == labels250[N:]) / len(outs)

#%%
# Save the trained model
torch.save(model.state_dict(), 'embed_model.pth')