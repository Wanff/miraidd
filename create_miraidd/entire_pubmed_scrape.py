import pandas as pd
from tqdm import tqdm
from datetime import date
import pickle
import numpy as np

from io import StringIO
from Bio import Entrez, Medline

from time import sleep

def get_paper_info(query: str, query_info: List, email:str):
    initiate = search_medline(query, email, retstart=0, retmax=100000, usehistory='y')
    rec_handler = search_medline(
        query, 
        email, 
        retstart=0, 
        retmax=100000, 
        usehistory='y',
        webenv=initiate['WebEnv'],
        query_key=initiate['QueryKey'],
    )
    
    papers_info = []
    try:
        for rec_id in rec_handler['IdList']:
            rec = fetch_rec(rec_id, rec_handler)
            rec_file = StringIO(rec)
            medline_rec = Medline.read(rec_file)
            # print(medline_rec)
            
            info = {}
            
            for i in query_info:
                if i in medline_rec:
                    info[i] = medline_rec[i]
                else:
                    raise Exception(f"{i} not in medline_rec")
            
            papers_info.append(info)
        
        return papers_info
    except:
        return 0
    
mirnas = pd.read_csv("../databases/external_databases/mirna.txt", sep="\t", header=None)

mirnas.columns = ['mir_num_id', 'mir_acc-id', 'mir', 'alt_mir', 'loop', 'seq', 'paper', 'irrelevant', 'irrelevant']

mirnas["loop"] = mirnas["loop"].fillna("")
#get rows that contain Homo Sapiens in the loop column of miRNAs
mirbase_mirs = mirnas[mirnas["loop"].str.contains("Homo sapiens")]['mir'].tolist()

mir_to_paper_info = {}

query_success = False
print("started")

for i, mir in tqdm(enumerate(mirbase_mirs)):
    spliced_mir = '-'.join(mir.split("-")[1:]) #gets rid of the hsa part
    query = f"({spliced_mir}[Title/Abstract]) AND (microRNA[MeSH Terms]) AND (Human[MeSH Terms])"

    print(mir)
    
    while not query_success:
        try:
            paper_info = get_paper_info(query, query_info = ["TI", "AB", "DP", "MH", "PMID"])
            print("success")
            query_success = True
        except:
            sleep(10)
            print("sleeping")
    
    mir_to_paper_info[mir] = paper_info
    
    query_success = False
    
    if i % 100 == 0:
        pickle.dump(mir_to_paper_info, open(f'entire_pmed_checkpoints/entire_pubmed_scrape_{i}.p', 'wb'))

pickle.dump(mir_to_paper_info, open('entire_pubmed_scrape.p', 'wb'))

#nohup python entire_pubmed_scrape.py &> entire_pubmed_scrape.out &

