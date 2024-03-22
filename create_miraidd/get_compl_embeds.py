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

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION_ID")

mesh_disesases = load_mesh_diseases()

#* loading pubmed

pubmed_w_compls = pd.read_csv("pubmed_w_compls_final.csv")

ada = OpenAIModel("text-embedding-ada-002")

embeds = []
for row_index, row in tqdm(list(pubmed_w_compls.iterrows())):
    print(row_index)
    if pd.isnull(row["completion"]) == True:
        print("Empty completion")
        print(row_index)
        print(row)
        print()
        continue
        
    while True:
        try:
            embed = ada.get_embedding(row["completion"])
            break
        except Exception as e:
            # If an exception is raised, print the error message and continue the loop
            print(f"Error: {e}")
            time.sleep(10)
            continue
    
    embeds.append((row_index, embed))
    
    if row_index % 100 == 0:
        print(f"Saving at {row_index}...")
        pickle.dump(embeds, open("pubmed_compl_embeds.p", "wb"))    

pickle.dump(embeds, open("pubmed_compl_embeds.p", "wb"))    

# nohup python get_compl_embeds.py &> get_compl_embeds.out &
