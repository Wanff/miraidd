import pandas as pd
from tqdm import tqdm
from datetime import date
from dotenv import load_dotenv
import os
import sys

import openai
import pickle

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from urllib.parse import quote

from lm_causal_utils import *

embeds = pickle.load(open("pubmed_compl_embeds.p", "rb"))

#embeds look like (index, embed). remove the embeds that have the same index
print(len(embeds))

import torch
import torch.nn as nn
import torch.optim as optim

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

model = MLP(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('embed_model.pth'))
print("model loaded")

# Evaluate the model on the test set
labels = []
with torch.no_grad():
    for index, embed in tqdm(embeds[:10]):
        outputs = model(torch.tensor(embed, dtype = torch.float32))
        
        labels.append(torch.round(outputs).item())

print(labels)
pickle.dump(labels, open("pubmed_compl_labels.p", "wb"))

#nohup python run_mlp.py &> mlp.out &