import pandas as pd
from tqdm import tqdm
from datetime import date
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import numpy as np

from pleiotropic_factors import *

%load_ext autoreload
%autoreload 2
# %%
mir_to_diseases = create_mir_to_diseases(disease_count=1)
print("num diseases created")

mir_to_num_diseases = {}

for m in mir_to_diseases:
    mir_to_num_diseases[m] = len(mir_to_diseases[m])

# %%
mirs_to_num = create_mir_to_num(mir_to_diseases.keys())
print("num created")
# %%
mir_to_num_papers = {}
pubmed = pd.read_csv("miRNA_databases/causal_pubmed.csv")

for idx, row in tqdm(pubmed.iterrows()):
    mir = row['mir']

    if mir in dead_mirs:
        continue

    if mir not in mir_to_num_papers.keys():
        mir_to_num_papers[mir] = 0
    
    mir_to_num_papers[mir] += 1
        
print("num papers created")

#%%
mir_to_avg_expression, mir_to_max_expression, mir_to_num_tissues = create_mir_to_expres(mir_to_diseases.keys())
print("expression created")
#%%
mir_to_conservation = create_mir_to_conservation(mir_to_diseases.keys())
print("conservation created")

#%%
for m in mir_to_diseases:
    mir_to_diseases[m] = "|".join(mir_to_diseases[m])

#%%
mir_to_miRbase_year = get_mir_to_miRbase_year(mir_to_diseases.keys())
#%%
mir_to_percent_cancer = {}

for m in mir_to_diseases:
    diseases = mir_to_diseases[m].split("|")
    
    num_cancer = 0
    for d in diseases:
        if is_disease_cancer(d):
            num_cancer += 1
    
    mir_to_percent_cancer[m] = num_cancer / len(diseases) if len(diseases) != 0 else 0
    
# %%
mir_to_diseases_df = pd.DataFrame.from_dict(mir_to_diseases, orient = "index", columns = ["diseases"])
mir_to_num_diseases_df = pd.DataFrame.from_dict(mir_to_num_diseases, orient = "index", columns = ["num_diseases"])
mir_to_num_papers_df = pd.DataFrame.from_dict(mir_to_num_papers, orient = "index", columns = ["num_papers"])
mirs_to_num_df = pd.DataFrame.from_dict(mirs_to_num, orient = "index", columns = ["num_in_name"])
mir_to_avg_expression_df = pd.DataFrame.from_dict(mir_to_avg_expression, orient = "index", columns = ["avg_expres"])
mir_to_max_expression_df = pd.DataFrame.from_dict(mir_to_max_expression, orient = "index", columns = ["max_expres"])
mir_to_conservation_df = pd.DataFrame.from_dict(mir_to_conservation, orient = "index", columns = ["conservation"])
mir_to_num_tissues_df = pd.DataFrame.from_dict(mir_to_num_tissues, orient = "index", columns = ["num_tissues"])
mir_to_miRbase_year_df = pd.DataFrame.from_dict(mir_to_miRbase_year, orient = "index", columns = ["miRbase_year"])
mir_to_percent_cancer_df = pd.DataFrame.from_dict(mir_to_percent_cancer, orient = "index", columns = ["percent_cancer"])

#%%
dfs = [mir_to_diseases_df, mir_to_num_diseases_df, mir_to_num_papers_df, mir_to_conservation_df, mirs_to_num_df, mir_to_avg_expression_df, mir_to_max_expression_df, mir_to_num_tissues_df, mir_to_miRbase_year_df, mir_to_percent_cancer_df]
pleiotropic_mirs = pd.concat(dfs, axis = 1, join='outer')
pleiotropic_mirs["mir"] = pleiotropic_mirs.index

#%%
miranda =  pd.read_csv("pleiotropic_mirs/pleiotropic_mirs_2023-05-08targets-miranda.csv")
mirdb = pd.read_csv("pleiotropic_mirs/pleiotropic_mirs_2023-05-08targets-mirdb", sep = ",")
targetscan = pd.read_csv("pleiotropic_mirs/pleiotropic_mirs_2023-05-08targets-scan.csv")

#merge mirdb, target_db, targetscan
target_db = pd.merge(miranda, targetscan)
pleiotropic_mirs = pd.merge(target_db, mirdb)

pleiotropic_mirs.drop(columns=["Unnamed: 0", "X"], inplace = True)

pleiotropic_mirs_cols = list(pleiotropic_mirs.columns)
pleiotropic_mirs_cols.remove("mir")
pleiotropic_mirs_cols.insert(0, "mir")

pleiotropic_mirs = pleiotropic_mirs[pleiotropic_mirs_cols]

pleiotropic_mirs.to_csv("pleiotropic_mirs/pleiotropic_mirs.csv", index=False)
