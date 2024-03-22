import pandas as pd
from tqdm import tqdm
from datetime import date
from dotenv import load_dotenv
import os
import textwrap
import sys
import time

import openai
from matplotlib import pyplot as plt
import plotly.express as px
import pickle

import numpy as np
import transformers
from sklearn.decomposition import PCA

import requests
from urllib.parse import quote

sys.path.append('/home/kevinrowang/miRNA-pleiotropy/')

from pleiotropic_factors import get_paper_info

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION_ID")

class OpenAIModel():
    def __init__(self, engine):
        self.engine = engine
    
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
                input = [text], 
                model=self.engine)['data'][0]['embedding']
    
    def get_chat_completion(self, messages, max_tokens: int = 1700):
        return openai.ChatCompletion.create(
            model=self.engine,
            messages=messages,
            max_tokens = max_tokens,
            )['choices'][0]['message']['content']
            
    def get_completion(self, text, max_tokens = 50):
        return openai.Completion.create(
            model=self.engine,
            prompt=[text],
            max_tokens=max_tokens,
            temperature=0
            )['choices'][0]['text']

def load_mesh_diseases(path = "../miRNA_databases/mesh_diseases_to_doid_dq.p"):
    mesh_diseases_to_doid = pickle.load(open(path, "rb"))

    mesh_diseases = []
    for d in mesh_diseases_to_doid:
        if not (mesh_diseases_to_doid[d]['doid_ancestor'] == '' and mesh_diseases_to_doid[d]['in_doid_exact'] == False):
            mesh_diseases.append(d)
    
    return mesh_diseases

def load_pubmed():
    entire_pubmed_og = pickle.load(open("../miRNA_databases/entire_pubmed_scrape.p", "rb"))
    entire_pubmed_no_papers = pickle.load(open("../miRNA_databases/entire_pubmed_scrape_no_paper_mirs.p", "rb"))

    entire_pubmed = entire_pubmed_og | entire_pubmed_no_papers

    mir_dfs =[]
    for mir in entire_pubmed.keys():
        mir_df = pd.DataFrame.from_dict(entire_pubmed[mir])
        mir_df["mir"] = mir
        mir_dfs.append(mir_df)

    pubmed = pd.concat(mir_dfs, ignore_index=True)
    
    return pubmed

def gen_chatgpt_completions(papers, chat_prompt, path_to_save = "", save_rate = 100, verbose = True):
    oai = OpenAIModel("gpt-3.5-turbo")
    w = textwrap.TextWrapper(width=75,break_long_words=False,replace_whitespace=False)

    completions = []
    for row_index, row in tqdm(list(papers.iterrows())):
        if pd.isnull(row["completion"]) == False:
            print(row["completion"])
            print()
            continue
        
        chat_prompt[1]["content"] = f"Title: {row['TI']}\nAbstract: {row['AB']}\nQuestion: Does {row['mir']} play a causal role in the disease described above?\nYour Answer:"
        
        while True:
            try:
                completion = oai.get_chat_completion(chat_prompt, max_tokens = 700)

                break

            except Exception as e:
                # If an exception is raised, print the error message and continue the loop
                print(f"Error: {e}")
                time.sleep(10)
                continue
        
        if verbose:
            # print(completion)
            # print()
            paper_data = w.fill(f"Title: {row['TI']}\nCompletion: {completion}\n\n")
            print(paper_data)
            print()
        
        
        completions.append(completion)
        
        papers.loc[row_index, "completion"] = completion
        
        if row_index % save_rate == 0 and path_to_save != "":
            print(f"Saving at {row_index}...")
            pickle.dump(completions, open(path_to_save, "wb"))
    
    if path_to_save != "":
        pickle.dump(completions, open(path_to_save, "wb"))
        
    papers.to_csv("pubmed_w_compls_final.csv")
    
    return completions
    
def hmdd_to_pubmed_df_format(hmdd):
    
    papers = []
    for row_index, row in tqdm(list(hmdd.iterrows())):
        
        try:
            paper = get_paper_info(row['pmid'], ["TI", "AB", "DP", "MH"])[0]
        except Exception as e:
            print(e)
            continue
            
        papers.append([paper['TI'], paper['AB'], paper['DP'], paper['MH'], row['pmid'], row['mir'], row['disease'],row['causality']])

    return pd.DataFrame(papers, columns = ["TI", "AB", "DP", "MH", "PMID", "mir", "disease", "causality"])

def pca_on_compl_embeds(embeds, n_components = 2):
    pca = PCA(n_components = n_components)

    pca.fit(embeds)
    
    pcs = []
    for i in range(n_components):
        pcs.append([np.dot(pca.components_[i], e) for e in embeds])
    return np.array(pcs)

def convert_pc_to_ans(pc):
    if pc < 0:
        return 1
    else:
        return 0

def pca_acc(pcs, labels, completions, papers, verbose = True):
    num_wrong = 0
    for i, pc, label, c, ab in zip(range(len(labels)), pcs, labels, completions, papers["AB"]):
        if convert_pc_to_ans(pc) != label:
            num_wrong += 1
    
    acc = 1 - (num_wrong / len(labels))
    
    if 1 - acc > acc:
        print(f"Accuracy (1-): {1 - acc}")
        for i, pc, label, c, ab in zip(range(len(labels)), pcs, labels, completions, papers["AB"]):
            if convert_pc_to_ans(pc) == label:
                print(f"index: {i}, my ans: {label}, pc: {pc}")
                print(textwrap.fill(c, 75)+ "\n")
                print(textwrap.fill(ab, 75)+ "\n\n")
                
        return 1 - acc
    else:
        print(f"Accuracy: {acc}")
        for i, pc, label, c, ab in zip(range(len(labels)), pcs, labels, completions, papers["AB"]):
            if convert_pc_to_ans(pc) != label:
                print(f"index: {i}, my ans: {label}, pc: {pc}")
                print(textwrap.fill(c, 75)+ "\n")
                print(textwrap.fill(ab, 75)+ "\n\n")
                
        return acc 
