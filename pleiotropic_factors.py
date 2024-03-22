import pandas as pd
from tqdm import tqdm
import re
from matplotlib import pyplot as plt
import plotly.graph_objects as go

import numpy as np
import plotly.express as px

import networkx as nx

from io import StringIO
from Bio import Entrez, Medline

from typing import List

dead_mirs = ['hsa-mir-547',
 'hsa-mir-219-2',
 'hsa-mir-801',
 'hsa-mir-720',
 'hsa-mir-886',
 'hsa-mir-3662',
 'hsa-mir-126a',
 'hsa-mir-9a',
 'hsa-mir-157',
 'hsa-mir-467a',
 'hsa-mir-192a',
 'hsa-mir-1165',
 'hsa-mir-218a',
 'hsa-mir-209',
 'hsa-mir-142a',
 'hsa-mir-2166',
 'hsa-mir-1280',
 'hsa-mir-112',
 'hsa-mir-161',
 'hsa-mir-101a',
 'hsa-mir-462a',
 'hsa-mir-741',
 'hsa-mir-1300',
 'hsa-mir-299a',
 'hsa-mir-1274a',
 'hsa-mir-388',
 'hsa-mir-189',
 'hsa-mir-219-1',
 'hsa-mir-5481',
 'hsa-mir-322',
 'hsa-mir-156a',
 'hsa-mir-690',
 'hsa-mir-768',
 'hsa-mir-712',
 'hsa-mir-1274b',
 'hsa-mir-145a',
 'hsa-mir-160',
 'hsa-mir-703',
 'hsa-mir-709',
 'hsa-mir-923',
 'hsa-mir-528a',
 'hsa-mir-743a',
 'hsa-mir-9501',
 'hsa-mir-21a',
 'hsa-mir-31a',
 'hsa-mir-786',
 'hsa-mir-1826',
 'hsa-mir-350',
 'hsa-mir-168',
 'hsa-mir-43c',
 'hsa-mir-192-2',
 'hsa-mir-705',
 'hsa-mir-696',
 'hsa-mir-191a',
 'hsa-mir-294',
 'hsa-mir-351',
 'hsa-mir-327',
 'hsa-mir-3546',
 'hsa-mir-1896',
 'hsa-mir-3591',
 'hsa-mir-719',
 'hsa-mir-1201',
 'hsa-mir-633b',
 'hsa-mir-5338',
 'hsa-mir-367a',
 'hsa-mir-1974',
 'hsa-mir-1308',
 'hsa-mir-565',
 'hsa-mir-694',
 'hsa-mir-326b',
 'hsa-mir-3098',
 'hsa-mir-223a',
 'hsa-mir-422b',
 'hsa-mir-181-2',
 'hsa-mir-3172',
 'hsa-mir-774',
 'Hsa-mir-93',
 'hsa-mir-648a',
 'hsa-mir-200C',
 'hsa-mir-3007a',
 'hsa-mir-220a',
 'hsa-mir-90b',
 'hsa-mir-1897',
 'hsa-mir-691',
 'hsa-mir-872',
 'hsa-mir-355',
 'hsa-mir-3560',
 'hsa-mir-3588',
 'hsa-mir-466b',
 'hsa-mir-6315',
 'hsa-mir-370a',
 'hsa-mir-467d',
 'hsa-mir-336',
 'hsa-mir-21b',
 'hsa-mir-7a']


def create_mir_to_diseases(path = 'databases/miraidd.csv', use_set: bool = True, disease_count: int = 1):
    """
    databases/miraidd.csv
    miRNA_databases/HMDD3_causal_info.csv
    
    Gets the number of diseases a miRNA is causal for in HMDD
    """
    is_hmdd = False
    if "HMDD" in path:
        print("HMDD")
        is_hmdd = True
        db = pd.read_csv(path, header = None, encoding= 'unicode_escape')

        db = db[[1, 2, 3, 5]].iloc[1:]
        db.columns = ['mir', 'disease', 'mesh_name', 'causal']
    else:
        db = pd.read_csv(path)
                
    mir_to_diseases = {}
    
    for idx, row in tqdm(db.iterrows()):
        mir = row['mir']
        
        if is_hmdd:
            diseases = row['disease']
            
        else:
            diseases = eval(row['diseases'])

        if mir in dead_mirs:
            continue
        
        if mir not in mir_to_diseases.keys():
            mir_to_diseases[mir] = []
        
        if row['causal'] == 'yes' or row['causal'] == 1.0:
            #in the case of causal_pubmed, this is a list of lists
            mir_to_diseases[mir].append(diseases)

    if use_set:
        for mir in mir_to_diseases.keys():
            if is_hmdd:
                mir_to_diseases[mir] = list(set(mir_to_diseases[mir]))
            else:
                diseases = [i for row in mir_to_diseases[mir] for i in row]
                
                disease_to_count = {}
                for d in diseases:
                    if d not in disease_to_count.keys():
                        disease_to_count[d] = 0
                    
                    disease_to_count[d] += 1
                
                print(mir)
                print(disease_to_count)
                diseases = [d for d in disease_to_count.keys() if disease_to_count[d] >= disease_count]
                mir_to_diseases[mir] = list(set(diseases))

    return mir_to_diseases

def mir_to_per_year_stats(path = 'databases/miraidd.csv', filter_causal: bool = True):
    db = pd.read_csv(path)
    
    mir_to_disease_and_year = {}

    for idx, row in tqdm(db.iterrows()):
        
        mir = row['mir']
        diseases = eval(row['diseases'])
        year = int(row['DP'][:4])

        if mir in dead_mirs:
            continue
        if mir not in mir_to_disease_and_year.keys():
            mir_to_disease_and_year[mir] = []
        
        if filter_causal:
            if row['causal'] == 1.0:
                mir_to_disease_and_year[mir].append((diseases, year))
        else:
            mir_to_disease_and_year[mir].append((diseases, year))

    return mir_to_disease_and_year

def mir_to_earliest_year(path = 'databases/miraidd.csv', filter_causal: bool = True):
    mir_to_disease_and_year = mir_to_per_year_stats(path = path, filter_causal = filter_causal)
    
    mir_to_earliest_year_dict = {}
    for mir in mir_to_disease_and_year:
        mir_to_earliest_year_dict[mir] = min([i[1] for i in mir_to_disease_and_year[mir]])

    return mir_to_earliest_year_dict

def create_mir_to_asso_diseases(HMDD_db_path = 'miRNA_databases/HMDD3_association_info.txt', use_set: bool = True):
    """
    Gets the number of diseases a miRNA is causal for in HMDD
    """
    HMDD_db = pd.read_csv(HMDD_db_path, header = None, sep="\t", encoding= 'unicode_escape')

    HMDD_db = HMDD_db[[1, 2, 3, 4]].iloc[1:]
    HMDD_db.columns = ['mir', 'disease', 'pmid', 'description']

    mir_to_diseases = {}

    for idx, row in tqdm(HMDD_db.iterrows()):
        mir = row['mir']
        disease = row['disease']

        if mir in dead_mirs:
            continue
        
        if mir not in mir_to_diseases.keys():
            mir_to_diseases[mir] = []
        
        mir_to_diseases[mir].append(disease)

    if use_set:
        for mir in mir_to_diseases.keys():
                mir_to_diseases[mir] = list(set(mir_to_diseases[mir]))
        
    return mir_to_diseases

def create_mir_to_pleiotropy(mir_to_diseases):
    mir_to_pleiotropy = {}
    
    for mir in mir_to_diseases.keys():
        mir_to_pleiotropy[mir] = len(list(set(mir_to_diseases[mir])))
    
    return mir_to_pleiotropy

def create_mir_to_num_dis_clusters(mir_to_diseases, miRNA_sim_adj_matrix_path = "STG_matrix_unpruned.csv"):
    #mir to num diseases according to the HMDD ontology clusters

    STG_matrix = pd.read_csv(miRNA_sim_adj_matrix_path)
    STG_matrix = STG_matrix.drop("Unnamed: 0", axis = 1)
    STG_matrix.index = STG_matrix.columns

    disease_names  = STG_matrix.columns

    STG_matrix = pd.DataFrame(data = np.where(STG_matrix < .7, 0, STG_matrix), index = disease_names, columns = disease_names)

    STG_nx = nx.from_pandas_adjacency(STG_matrix, create_using = nx.DiGraph)

    dis_clusters = list(nx.strongly_connected_components(STG_nx))

    mir_to_num_dis_clusters = {}

    for mir in tqdm(mir_to_diseases.keys()):
        num = 0

        for cluster in dis_clusters:
            if len(cluster.intersection(set(mir_to_diseases[mir]))) >= 1:
                num += 1
        
        mir_to_num_dis_clusters[mir] = num

    return mir_to_num_dis_clusters

def create_mir_to_num(mirs: List[str]):
    mirs_to_num = {}

    for mir in mirs:
        num = mir.split("-")[2]
        mirs_to_num[mir] = int(re.sub('[^0-9]','', num))
    
    return mirs_to_num

#PUBMED functions
def search_medline(query, email,**kwargs):
    Entrez.email = email
    search = Entrez.esearch(db='pubmed', term=query, **kwargs)
    try:
        handle = Entrez.read(search)
        return handle
    except RuntimeError as e:
        return None
    except Exception as e:
        print(e)
        # raise IOError(str(e))
        return None
    finally:
        search.close()

def fetch_rec(rec_id, entrez_handle):
    fetch_handle = Entrez.efetch(db='pubmed', id=rec_id,
                                 rettype='Medline', retmode='text',
                                 webenv=entrez_handle['WebEnv'],
                                 query_key=entrez_handle['QueryKey'])
    rec = fetch_handle.read()
    return rec

def tryreturn(record, col):
    try: 
        if (col == 'FAU') or (col == 'AD'):
            return '; '.join(record[col])
        elif col=='COIS':
            return ' '.join(record[col])
        else:
            return record[col]
    except:
        return ''

#deprecated
def get_num_papers(query, email, return_paper_dates = False):
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
            
    if return_paper_dates:
        date_to_count = {}
        for rec_id in rec_handler['IdList']:
            rec = fetch_rec(rec_id, rec_handler)
            rec_file = StringIO(rec)
            medline_rec = Medline.read(rec_file)
            
            # print(medline_rec)
            if 'DP' in medline_rec:
                date = int(medline_rec['DP'][:4])
                #extract int from date
                # date = int(re.sub('[^0-9]','', date))
                
                if date not in date_to_count:
                    date_to_count[date] = 0
                
                date_to_count[date] +=1 
            
        return int(initiate['Count']), date_to_count
    else:
        return int(initiate['Count'])

def get_paper_year(query, email = "kevn.wanf@gmail.com"):
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
    
    
    try:
        for rec_id in rec_handler['IdList']:
            rec = fetch_rec(rec_id, rec_handler)
            rec_file = StringIO(rec)
            medline_rec = Medline.read(rec_file)
            
            # print(medline_rec)
            if 'DP' in medline_rec:
                date = int(medline_rec['DP'][:4]) 
        
        return date
    except:
        return 0

def get_paper_info(query: str, query_info: List, email:str = "kevn.wanf@gmail.com"):
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
                    info[i] = ''
            
            papers_info.append(info)
        
        return papers_info
    except:
        return 0
    
def create_mir_to_num_papers(mirs: List[str], email =  "kevn.wanf@gmail.com", return_paper_dates = False):
    mir_to_num_papers = {}
    mir_to_paper_dates = {}
    for mir in tqdm(mirs):
        spliced_mir = '-'.join(mir.split("-")[1:]) #gets rid of the hsa part
        query = f"({spliced_mir}[Title/Abstract]) AND (microRNA[MeSH Terms]) AND (Human[MeSH Terms] OR Mouse[MeSH Terms])"
        
        if return_paper_dates is False:
            mir_to_num_papers[mir] = get_num_papers(query, email, return_paper_dates = return_paper_dates)
        else:
            mir_to_num_papers[mir], mir_to_paper_dates[mir] = get_num_papers(query, email, return_paper_dates = return_paper_dates)
    
    if return_paper_dates is False:
        return mir_to_num_papers
    else:
        return mir_to_num_papers, mir_to_paper_dates

def create_mir_to_expres(mirs, miRmine_path = "databases/external_databases/miRmine/miRmine-tissues.csv"):
    miRmine = pd.read_csv(miRmine_path)

    new_mirs_names = []
    for mir in np.array(miRmine['Precursor miRNA ID']):
        #filters mir to hsa-mir-number
        # if len(mir.split("-")) > 3:
            # mir = '-'.join(mir.split("-")[:-1]) 
        # if mir[-1].isalpha():
            # mir = mir[:-1]
        new_mirs_names.append(mir)

    miRmine.insert(loc = 2, column = "Filtered miRNA ID", value = new_mirs_names)

    mir_to_avg_expression = {}
    mir_to_max_expression = {}
    mir_to_num_tissues = {}
    
    tissue_to_mirmine_idx = {
        'Lung': [],
        'Saliva': [],
        'Bladder': [],
        'Blood': [],
        'Hair follicle': [],
        'Plasma': [],
        'Placenta': [],
        'Serum': [],
        'Testis': [],
        'Pancreas': [],
        'Nasopharynx': [],
        'Liver': [],
        'Brain': [],
        'Breast': [],
        'Semen': [],
        'Sperm': [],
    }

    for tissue in tissue_to_mirmine_idx:
        for i, mirmine_tissue in enumerate(miRmine.columns.values):
            if tissue in mirmine_tissue:
                tissue_to_mirmine_idx[tissue].append(i)
    
    for og_mir in mirs:
        mir = og_mir
        # if len(mir.split("-")) > 3:
        #     mir = '-'.join(mir.split("-")[:-1]) 
        # if mir[-1].isalpha():
        #     mir = mir[:-1]
        
        tissue_count = 0
        for tissue in tissue_to_mirmine_idx:
            tissue_expres = np.array(miRmine[miRmine['Filtered miRNA ID'] == mir])[:, tissue_to_mirmine_idx[tissue]].flatten()
            
            if np.average(tissue_expres) > 0:
                tissue_count += 1 
        
        mir_to_num_tissues[og_mir] = tissue_count
        
        exprsn_values = np.array(miRmine[miRmine['Filtered miRNA ID'] == mir])[:,3:].flatten()

        exprsn_values = exprsn_values.astype("float")
        
        try:
            mir_to_max_expression[og_mir] = max(exprsn_values)
            mir_to_avg_expression[og_mir] = np.average(exprsn_values)
        except:
            mir_to_max_expression[og_mir] = 0
            mir_to_avg_expression[og_mir] = 0
    
    return mir_to_avg_expression, mir_to_max_expression, mir_to_num_tissues


#pretty faulty
def get_miRbase_id(og_mir, path = "miRNA_databases/"):
    mirnas = pd.read_csv(path + "mirna.txt", sep="\t", header = None)
    mirnas.columns = ['mir_num_id', 'mir_acc-id', 'mir', 'alt_mir', 'loop', 'seq', 'paper', 'irrelevant', 'irrelevant']
    
    mirnas.replace(np.nan, '', inplace=True)
    
    mir = re.sub('(-5p|-3p|.3p|.5p)$', '', og_mir)

    #Gets the miRbase ID of the miRNA
    mir_num_id = mirnas[mirnas['mir'].str.contains(mir)].mir_num_id.values
    if len(mir_num_id) == 0: #check if alt_mir contains match
        mir_num_id = mirnas[mirnas['alt_mir'].str.contains(mir)].mir_num_id.values

    if len(mir_num_id) == 0: 
        if len(mir.split("-")) > 3:
            mir = '-'.join(mir.split("-")[:-1]) 
        if mir[-1].isalpha():
            mir = mir[:-1]

        mir_num_id = mirnas[mirnas['mir'].str.contains(mir)].mir_num_id.values
        if len(mir_num_id) == 0: #check if alt_mir contains match
            mir_num_id = mirnas[mirnas['alt_mir'].str.contains(mir)].mir_num_id.values
            
           
        
    # if len(mir_num_id.index) == 0: #now check if str.contains works
    #     regex = r'^' + re.escape(mir) + '([a-z]|-\d|[a-z]-\d)$' 
    #     mir_num_id_idx = mirnas['mir'].loc[lambda x: x.str.match(regex, case=False, na=False)].index
    #     mir_num_id_idx.append(mirnas['alt_mir'].loc[lambda x: x.str.match(regex, case=False, na=False)].index)
    #     if len(mir_num_id.index) == 0:
    #         mir_num_id = 0.0
    #     else:
    #         mir_num_id = mirnas.iloc[mir_num_id_idx[0]].mir_num_id
    
    
    if len(mir_num_id) == 0:
        return 0
    else:
        return mir_num_id[0]

def create_mir_to_conservation(miRNA_names, path = "miRNA_databases/"):
    mirna_fam_map = pd.read_csv(path + "mirna_2_prefam.txt", sep="\t", header=None)
    mirnas = pd.read_csv(path + "mirna.txt", sep="\t", header=None)

    mirna_fam_map.columns = ['mir_num_id', 'fam_num_id']
    mirnas.columns = ['mir_num_id', 'mir_acc-id', 'mir', 'alt_mir', 'loop', 'seq', 'paper', 'irrelevant', 'irrelevant']

    fam_dict = {}
    for og_mir in tqdm(miRNA_names):
        mir_num_id = get_miRbase_id(og_mir)

        fam_num_id = mirna_fam_map[mirna_fam_map['mir_num_id']==mir_num_id].fam_num_id
        if len(fam_num_id.index) == 0:
            fam_num_id = 0
        fam_num_id = int(fam_num_id)

        num_fam_membs = len(mirna_fam_map[mirna_fam_map['fam_num_id']==fam_num_id].index) #rowcount for how many members of that family
        fam_dict[og_mir] = num_fam_membs

    return fam_dict

def get_mir_to_miRbase_year(miRNA_names, path = "miRNA_databases/"):
    mirna_lit_refs = pd.read_csv(path + "mirna_literature_references.txt", sep="\t", header=None)
    lit_refs = pd.read_csv(path + "literature_references.txt", sep="\t", header=None)
    
    mirna_lit_refs.columns = ['mir_id', 'lit_ref_id', 'nan',  'paper_idx']
    lit_refs.columns = ['lit_ref_id', 'nan', 'title', 'authors', 'journal_year']
    
    mir_to_miRbase_year = {}
    for og_mir in tqdm(miRNA_names):
        mir_num_id = get_miRbase_id(og_mir) 
        # print(mir_num_id)
        lit_ref_ids = mirna_lit_refs[mirna_lit_refs['mir_id']==mir_num_id].lit_ref_id.values       
        lit_ref_ids = [str(x) for x in lit_ref_ids]
        
        # print(lit_ref_ids)
        journal_years = lit_refs[lit_refs['lit_ref_id'].isin(lit_ref_ids)].journal_year
        # print(journal_years)
        journal_years = [int(year[-6:-2]) for year in journal_years]
        
        mir_to_miRbase_year[og_mir] = min(journal_years) if len(journal_years) > 0 else 0
    
    return mir_to_miRbase_year

def plot_pleiotropy(pleiotropic_mirs_og, x_axis, y_axis, log_scale_x = True, log_scale_y = True, color = "num_papers", log_scale_num_papers = False, hover_data = ['name', 'num_diseases', 'num_papers',  'num_in_name', 'max_expres', 'avg_expres', 'conservation', 'num_targets'], draw_y_equals_x = False):
    pleiotropic_mirs  = pleiotropic_mirs_og.copy(deep = True)
    if log_scale_x:
        pleiotropic_mirs[x_axis] = np.log2(pleiotropic_mirs[x_axis] + 1) + 1
    
    if log_scale_y:
        pleiotropic_mirs[y_axis] = np.log2(pleiotropic_mirs[y_axis] + 1) + 1

    if log_scale_num_papers:
        pleiotropic_mirs['num_papers'] = np.log2(pleiotropic_mirs['num_papers'] + 1) + 1

    fig = px.scatter(pleiotropic_mirs, x = x_axis, y = y_axis, hover_data= hover_data, color = color)
    
    if draw_y_equals_x:
        fig.add_shape(
            type="line",
            x0=fig.data[0].x.min(),
            y0=fig.data[0].x.min(),
            x1=fig.data[0].x.max(),
            y1=fig.data[0].x.max(),
            line=dict(color="red", width=1, dash="dash"),
        )
        
    fig.show()
    # print(pearsonr(x, y))

def plot_pleiotropy_3d(pleiotropic_mirs_og, x_axis, y_axis, z_axis, log_scale_x = True, log_scale_y = True, log_scale_z = True):
    pleiotropic_mirs  = pleiotropic_mirs_og.copy(deep = True)
    
    if log_scale_x:
        pleiotropic_mirs[x_axis] = np.log2(pleiotropic_mirs[x_axis]) + 1
    
    if log_scale_y:
        pleiotropic_mirs[y_axis] = np.log2(pleiotropic_mirs[y_axis]) + 1
    
    if log_scale_z:
        pleiotropic_mirs[z_axis] = np.log2(pleiotropic_mirs[z_axis]) + 1

    fig = px.scatter_3d(pleiotropic_mirs, x = x_axis, y = y_axis, z= z_axis, hover_data= ['name', 'num_diseases', 'num_papers',  'num_in_name', 'max_expres', 'avg_expres', 'conservation', 'num_targets'], size  = "num_diseases", size_max = 18)
    fig.show()


    
def is_disease_cancer(disease):
        if "carcinoma" in disease.lower() or "neoplasms" in disease.lower() or "cancer" in disease.lower() or "tumor" in disease.lower() or "leukemia" in disease.lower() or "lymphoma" in disease.lower() or "polycythemia vera" in disease.lower() or "blastoma" in disease.lower() or "sarcoma" in disease.lower():
            return True
        else:
            return False   

def create_mir_to_percent_cancer(HMDD_db_path = 'miRNA_databases/HMDD3_causal_info.csv'):
    mir_to_diseases = create_mir_to_diseases()
    
    mir_to_percent_cancer = {}
    
    for mir in mir_to_diseases.keys():
        cancers = []
        for disease in mir_to_diseases[mir]:
            if is_disease_cancer(disease):
                cancers.append(disease)
        
        mir_to_percent_cancer[mir] = len(cancers) / len(mir_to_diseases[mir]) if len(mir_to_diseases[mir]) != 0 else 0
    
    return mir_to_percent_cancer

def mir_to_diseases_per_year(HMDD_db_path = 'miRNA_databases/HMDD3_causal_info.csv'):
    HMDD_db = pd.read_csv(HMDD_db_path, header = None, encoding= 'unicode_escape')

    HMDD_db = HMDD_db[[1, 2, 3, 4, 5]].iloc[1:] #gets rid of header row and category column
    HMDD_db.columns = ['mir', 'disease', 'mesh_name', 'pmid', 'causality']

    mir_to_papers = {}

    for idx, row in tqdm(HMDD_db.iterrows()):
        mir = row['mir']
        disease = row['disease']

        if mir not in mir_to_papers.keys():
            mir_to_papers[mir] = []
        
        if row['causality'] == 'yes':
            mir_to_papers[mir].append(row['pmid'])

    mir_to_paper_years = {}
    
    for mir in tqdm(list(mir_to_papers.keys())):
        pmids = mir_to_papers[mir]
        
        mir_to_paper_years[mir] = [get_paper_year(pmid) for pmid in pmids]

        print(mir_to_paper_years)
    
    return mir_to_paper_years


def calc_mir_trends(plot_mir, causal_pubmed, total_num_diseases = 6461):
    # Group the data by year and calculate the fraction of causal papers per year
    grouped_data = causal_pubmed[causal_pubmed.mir == plot_mir].groupby('year')['causal'].mean().reset_index() 
    grouped_data['causal_fraction'] = grouped_data['causal'] * 100  

    #*new diseases per year
    mir_to_disease_and_year = mir_to_per_year_stats(filter_causal = True)

    mir_to_new_disease_and_year = {}

    for mir in mir_to_disease_and_year:
        mir_to_new_disease_and_year[mir] = {}
        
        sorted_years = sorted(mir_to_disease_and_year[mir], key = lambda x: x[1])
        
        mirs_diseases = []
        for diseases, year in sorted_years:
            
            if year not in mir_to_new_disease_and_year[mir]:
                mir_to_new_disease_and_year[mir][year] = []
            
            for disease in diseases:
                if disease not in mirs_diseases:
                    mir_to_new_disease_and_year[mir][year].append(disease)
                    mirs_diseases.append(disease)

    data_dict = mir_to_new_disease_and_year[plot_mir].copy()
    for year in data_dict:
        data_dict[year] = len(data_dict[year])
            
    years = list(data_dict.keys())
    values = list(data_dict.values())

    papers_per_year = causal_pubmed[causal_pubmed.mir == plot_mir]["year"].value_counts().tolist()
    grouped_data['papers_per_year'] = papers_per_year[::-1]
    grouped_data['causal_papers_per_year'] = [np.round((frac_causal / 100) * paper) for paper, frac_causal in zip(grouped_data['papers_per_year'], grouped_data['causal'])]

    actual_years = sorted(causal_pubmed[causal_pubmed.mir == plot_mir]["year"].value_counts().index.tolist())

    # #there can be a mismatch between actual_years and years because years only keeps track of the years that have new diseases. we fix this:

    for year in actual_years:
        if year not in years:
            index = actual_years.index(year)
            years.insert(index, year)
            values.insert(index, 0)
    grouped_data['new_dis_per_year'] = values 

    fraction_new_dis_per_year = []
    for year, num_new_dis, num_papers in zip(years, values, papers_per_year[::-1]):
        fraction_new_dis_per_year.append((100 * num_new_dis)/num_papers)
    grouped_data['frac_new_dis_per_year'] = fraction_new_dis_per_year 



    frac_total_dis_per_year = []
    for i in range(len(values)):
        frac_total_dis_per_year.append((sum(values[:i+1]) / total_num_diseases)*100 )
    grouped_data['frac_total_dis_per_year'] = frac_total_dis_per_year

    return grouped_data

def plot_mir_trends(plot_mir, grouped_data, causal_pubmed):
    histogram_fig = px.histogram(
    causal_pubmed[causal_pubmed.mir == plot_mir],
    x="year",
    color="causal",
    category_orders=dict(year=list(range(2004, 2024))),
    title="% Causal miRNA papers per year hsa-mir-21"
    )

    # Create the bar chart
    hist_fig_new_dis = go.Figure(data=go.Bar(x=grouped_data['year'], y=grouped_data['new_dis_per_year']))
        
    # Create the line plot
    line_fig = go.Figure()
    line_fig.add_trace(
        go.Scatter(
            x=grouped_data['year'],
            y=grouped_data['causal_fraction'],
            mode='lines',
            name='% Causal Papers',
            yaxis='y2'
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=grouped_data['year'],
            y=grouped_data['frac_new_dis_per_year'],
            mode='lines',
            name='% New Disease Papers',
            yaxis='y2'
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=grouped_data['year'],
            y=grouped_data['frac_total_dis_per_year'],
            mode='lines',
            name='% of Total Disease caused',
            yaxis='y2'
        )
    )

    line_fig.update_layout(
        title="Fraction of Causal Papers per Year for "+plot_mir,
        yaxis=dict(title='Number of Papers'),
        yaxis2=dict(
            title='Fraction of Causal Papers (%)',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        font_family = "Times New Roman"
    )

    # Combine the histogram and line plot
    combined_fig = go.Figure(data=histogram_fig.data + line_fig.data + hist_fig_new_dis.data, layout=line_fig.layout)

    combined_fig.update_layout(showlegend=False)
    # Display the combined plot
    combined_fig.show()


# %%
