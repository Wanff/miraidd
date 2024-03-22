#%%
import pandas as pd
from tqdm import tqdm
from datetime import date
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import requests
from pleiotropic_factors import *

%load_ext autoreload
%autoreload 2

#%%
#* LOAD DATA
pleiotropic_mirs = pd.read_csv("databases/pleiotropic_mirs.csv")

causal_pubmed = pd.read_csv("databases/miraidd.csv")

causal_pubmed["year"] = pd.to_numeric(causal_pubmed["DP"].str[:4])

#%%

pleiotropic_mirs.drop(columns = ['num_in_name'], inplace = True)

#%%
#* Increase in papers per year
mir_to_disease_and_year = mir_to_per_year_stats(filter_causal = False)

all_years = [mir_to_disease_and_year[mir] for mir in mir_to_disease_and_year.keys()]
all_years = [item[1] for sublist in all_years for item in sublist]

#convert all_years to a dict that maps years to how many times it appears in the list

# year_to_num_papers = {}
# for year in all_years:
#     if year not in year_to_num_papers.keys():
#         year_to_num_papers[year] = 0
#     year_to_num_papers[year] += 1

# year_to_num_papers_df = pd.DataFrame.from_dict(year_to_num_papers, orient='index', columns = ['Number of Papers'])
  
#%%
#remove all occurrences of 2023 in all_years
all_years = np.array(all_years)
all_years = all_years[all_years != 2023]

2023 in all_years        
# %%
fig = px.histogram(all_years, title = "Number of Published miRNA papers per Year",color_discrete_sequence = ["#001F3F"])
fig.update_layout(
    font_family="Times New Roman",
    yaxis_title="Number of Papers",
    xaxis_title = "Year"
)
fig.show()
# %%
mir_to_causal_disease_and_year = mir_to_per_year_stats(filter_causal = True)

all_years_c = [mir_to_causal_disease_and_year[mir] for mir in mir_to_causal_disease_and_year.keys()]
all_years_c = [item[1] for sublist in all_years_c for item in sublist]


all_years_c = np.array(all_years_c)
all_years_c = all_years_c[all_years_c != 2023]

#%%     
fig = px.histogram(all_years_c, title = "Number of Causal miRNA papers per Year",color_discrete_sequence = ["#001F3F"])
fig.update_layout(
    font_family="Times New Roman",
    yaxis_title="Number of Causal Papers",
    xaxis_title = "Year"
)
fig.show()

#%%
#* scatter plot of num_papers and num_diseases
pleiotropic_mirs_copy  = pleiotropic_mirs.copy(deep = True)
pleiotropic_mirs_copy["num_papers"] = np.log2(pleiotropic_mirs_copy["num_papers"] + 1) + 1

pleiotropic_mirs_copy["num_diseases"] = np.log2(pleiotropic_mirs_copy["num_diseases"] + 1) + 1

fig = px.scatter(pleiotropic_mirs_copy, x="num_papers", y="num_diseases", color_discrete_sequence=["#004141"] )

# Add the R-squared value as text annotation

fig.update_layout(
        font_family="Times New Roman",
    xaxis_title="Number of Papers (Log 2)",
    yaxis_title="Number of Diseases (Log 2)",
)

fig.show()
#%%
X_val = "num_in_name"
Y_val = "max_expres"

pleiotropic_mirs_copy  = pleiotropic_mirs.copy(deep = True)
pleiotropic_mirs_copy[X_val] = np.log2(pleiotropic_mirs_copy[X_val] + 1) + 1

pleiotropic_mirs_copy[Y_val] = np.log2(pleiotropic_mirs_copy[Y_val] + 1) + 1

fig = px.scatter(pleiotropic_mirs_copy, x=X_val, y=Y_val, color_discrete_sequence=["#004141"] )

# Add the R-squared value as text annotation

fig.update_layout(
    title = "Expression Value vs Number in miRNA Name",
    font_family="Times New Roman",
    xaxis_title="Number in Name (Log 2)",
    yaxis_title="Max Expression (Log 2)",
)

fig.show()


#%%
#* CORRELATION MATRICES
from pingouin import partial_corr

# pleiotropic_mirs = pd.read_csv("databases/pleiotropic_mirs.csv")
#%%
pleiotropic_mirs.drop(columns = ["years_resids","paper_resids", "full_lr_pred", "num_diseases_thresh_2", "no_paper_lr_pred"], inplace = True)
#%%
pleiotropic_mirs
#%%

pleiotropic_mirs = pleiotropic_mirs[pleiotropic_mirs.miRbase_year != 0]
years = 2023- pleiotropic_mirs["miRbase_year"]
years[years == 2023] = 0
years = years.to_numpy()

# pleiotropic_mirs['miRbase_year'] = years
partial_corr_mat = []
partial_corr_col = "num_papers"
for x in pleiotropic_mirs.columns[2:]:
    partial_corr_mat_x = []
    for y in pleiotropic_mirs.columns[2:]:
        if x != y and x != partial_corr_col and y != partial_corr_col:
            pc_df = partial_corr(data=pleiotropic_mirs, x=x, y=y, covar=[partial_corr_col], method='pearson')
            
            print(pc_df)
            partial_corr_mat_x.append(pc_df['r'].values[0])
        else:
            partial_corr_mat_x.append(0)
    partial_corr_mat.append(partial_corr_mat_x)
#%%
partial_corr_mat = pd.DataFrame(partial_corr_mat, columns = pleiotropic_mirs.columns[2:], index = pleiotropic_mirs.columns[2:])

#%%
corr_mat = pleiotropic_mirs.iloc[:, 2:].corr()
#%%
import scipy
import scipy.cluster.hierarchy as sch
def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

#%%
column_names = f"Papers, Date Discovered, Targetscan Targets, Diseases, MirDB Targets, Conservation, Tissues present, % Cancer, Validated Targets, miRanda Targets, Max Expression Value, Average Expression Value, Number in miRNA name".split(", ")

col_to_fig_name = {
    "avg_expres": "Avg Expres Value",
    "max_expres": "Max Expres Value",
    "num_diseases":"Pleiotropy",
    "num_papers":"Papers",
    "conservation":"Conservation",
    "num_tissues":"Tissues present",
    "percent_cancer":"% Cancer",
    "num_targets":"Validated Targets",
    "miranda_targets":"miRanda",
    "mirdb_targets":"MirDB",
    "miRbase_year":"Date Discovered",
    "targetscan_targets":"Targetscan",
    "num_in_name":"Num in miR name"
}

def index_to_fig_name_list(index):
    return [col_to_fig_name[col] for col in index]

def imshow_df_w_all_tick_vals(df, title = "", column_names = []):
    fig = px.imshow(df.values, color_continuous_midpoint=0, color_continuous_scale='RdBu', title = title)

    # Set the x-axis and y-axis tick values
    fig.update_xaxes(tickvals=list(range(len(df.columns))), ticktext=index_to_fig_name_list(df.columns))
    fig.update_yaxes(tickvals=list(range(len(df.index))), ticktext=index_to_fig_name_list(df.index))

    # Display the plot
    fig.show()

imshow_df_w_all_tick_vals(cluster_corr(partial_corr_mat), title = "Partial Correlation Matrix Adjusting for Papers", column_names = column_names)
#%%
# Rearrange columns of corr_mat to match partial_corr_mat
corr_mat_rearranged = corr_mat[cluster_corr(partial_corr_mat).columns]
corr_mat_rearranged = corr_mat_rearranged.loc[cluster_corr(partial_corr_mat).index]

#%%
# Plot the rearranged correlation matrix
imshow_df_w_all_tick_vals(corr_mat_rearranged, title="Correlation Matrix")
# imshow_df_w_all_tick_vals(cluster_corr(corr_mat), title = "Correlation Matrix")

#%%
#*fraction causal papers per year
#%%
#* total num of diseases
all_diseases = [eval(l) for l in causal_pubmed["diseases"]] 
all_diseases = [item for sublist in all_diseases for item in sublist]

total_num_diseases = len(set(all_diseases))
total_num_diseases 

#%%
query = "disease"
api_url = "http://www.ebi.ac.uk/ols/api/search?q=" + query + "&ontology=doid&exact=True&allChildrenOf="+query

response = requests.get(api_url)

# response.json()
# response.json()['response']['docs']
# for term in response.json()['response']['docs']:
#     if term['is_defining_ontology'] == True:
#         self.doid_terms.append(query)
#         return True

# return False

#%%


#%%

query = "disease"
api_url = "https://www.ebi.ac.uk/ols4/api/ontologies/DOID/terms?obo_id=DOID%3A4&lang=en&page=0&size=1&sort=string&forceFirstAndLastRels=true"

def get_child_doids(query_doid):
    query_doid = query_doid.replace(":", "_")
    # api_url = "https://www.ebi.ac.uk/ols4/api/ontologies/DOID/terms?obo_id="+query_doid+"&lang=en&page=0&size=1&sort=string&forceFirstAndLastRels=true"
    api_url = f"https://www.ebi.ac.uk/ols4/api/ontologies/doid/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{query_doid}/children"
    
    response = requests.get(api_url)
    
    if response.status_code == 200:
        children = response.json()['_embedded']['terms']
        
        doid_and_children_status = [(child['short_form'], child['has_children']) for child in children]
        
        return doid_and_children_status
    else:
        raise ValueError


def count_leaf_nodes(query_doid, doids_visited, leaf_nodes):
    doids_visited.append(query_doid)
    
    child_doids = get_child_doids(query_doid)
    
    for child_doid in child_doids:
        doid = child_doid[0]
        children_status = child_doid[1]
        if doid not in doids_visited:
            if children_status is True:
                count_leaf_nodes(doid, doids_visited, leaf_nodes)
            else:
                leaf_nodes.append(doid)

leaf_nodes = []
query_doid = "DOID_4"
doids_visited = []

count_leaf_nodes(query_doid, doids_visited, leaf_nodes)
        
#%%
#save leaf_nodes to pickle

#%%
len(leaf_nodes)


#%%
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

plot_mir = "hsa-mir-21"
# Group the data by year and calculate the fraction of causal papers per year
grouped_data =  causal_pubmed[causal_pubmed.mir == plot_mir].groupby('year')['causal'].mean().reset_index()
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
#%%

papers_per_year = causal_pubmed[causal_pubmed.mir == plot_mir]["year"].value_counts().tolist()

actual_years = sorted(causal_pubmed[causal_pubmed.mir == plot_mir]["year"].value_counts().index.tolist())

# #there can be a mismatch between actual_years and years because years only keeps track of the years that have new diseases. we fix this:

for year in actual_years:
    if year not in years:
        index = actual_years.index(year)
        years.insert(index, year)
        values.insert(index, 0)

fraction_new_dis_per_year = []
for year, num_new_dis, num_papers in zip(years, values, papers_per_year[::-1]):
    fraction_new_dis_per_year.append((100 * num_new_dis)/num_papers)
grouped_data['new_ dis_per_year'] = fraction_new_dis_per_year 

frac_total_dis_per_year = []
for i in range(len(values)):
    frac_total_dis_per_year.append((sum(values[:i+1]) / total_num_diseases)*100 )
grouped_data['frac_total_dis_per_year'] = frac_total_dis_per_year

#%%

histogram_fig = px.histogram(
    causal_pubmed[causal_pubmed.mir == plot_mir],
    x="year",
    color="causal",
    category_orders=dict(year=list(range(2004, 2024))),
    title="% Causal miRNA papers per year hsa-mir-21"
)

# data_dict = mir_to_new_disease_and_year[plot_mir].copy()
# for year in data_dict:
#     data_dict[year] = len(data_dict[year])
        
# years = list(data_dict.keys())
# values = list(data_dict.values())

# # Create the bar chart
# hist_fig_new_dis = go.Figure(data=go.Bar(x=years, y=values))
    
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
        y=grouped_data['new_dis_per_year'],
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
combined_fig = go.Figure(data=histogram_fig.data + line_fig.data, layout=line_fig.layout)

combined_fig.update_layout(showlegend=False)
# Display the combined plot
combined_fig.show()



#%%
#* is there an overall decrease in causal papers per year?

num_disease_thresh = pleiotropic_mirs["num_diseases"].quantile(0.95) 
num_papers_thresh = pleiotropic_mirs["num_papers"].quantile(0.95)

most_causal_mirs = pleiotropic_mirs[ (pleiotropic_mirs["num_diseases"] >= num_disease_thresh) & (pleiotropic_mirs["num_papers"] >= num_papers_thresh)]

most_causal_mirs

#%%
most_causal_mirs.index.shape

#%%
from pleiotropic_factors import calc_mir_trends, plot_mir_trends

#%%
mir_to_mir_trends = {}
for mir in tqdm(most_causal_mirs["mir"]):
    mir_to_mir_trends[mir] = calc_mir_trends(mir, causal_pubmed)
#%%
with open("mir_to_mir_trends.p", "wb") as f:
    pickle.dump(mir_to_mir_trends, f)
    
#%%


#%%
for mir in most_causal_mirs["mir"]:
    plot_mir_trends(mir, mir_to_mir_trends[mir], causal_pubmed)

#%%
peak_years = []
for mir in most_causal_mirs["mir"]:
    max_frac = mir_to_mir_trends[mir]["frac_new_dis_per_year"].max()

    mir_perc_new_dis_peak_year = mir_to_mir_trends[mir][mir_to_mir_trends[mir]["frac_new_dis_per_year"] == max_frac]['year'].values[0]
    
    peak_years.append(mir_perc_new_dis_peak_year)

px.histogram(peak_years)

#%%
px.histogram(pleiotropic_mirs['num_diseases'])

#%%
px.histogram(np.log2(pleiotropic_mirs[pleiotropic_mirs['num_diseases'] > 10]['num_diseases']))

#%%
px.histogram(pleiotropic_mirs[pleiotropic_mirs['num_diseases'] > 10]['num_diseases'])

#%%
most_causal_mirs
#%%

#for each mir in most_causal_years, fit a line starting from the peak year to the last year for frac_new_dis_per_year and see if slope is negative
import scipy
slopes =[]
rs = []
ps = []
cepts = []
for mir in most_causal_mirs["mir"]:
    
    max_frac = mir_to_mir_trends[mir]["frac_new_dis_per_year"].max()

    mir_perc_new_dis_peak_year = mir_to_mir_trends[mir][mir_to_mir_trends[mir]["frac_new_dis_per_year"] == max_frac]['year'].values[0]
    
    mir_trends = mir_to_mir_trends[mir][(mir_to_mir_trends[mir]["year"] >= 2012) & (mir_to_mir_trends[mir]['year'] <= 2020)]
    
    # mir_trends = mir_trends[mir_trends["frac_new_dis_per_year"] > 0]
    
    if len(mir_trends) < 2:
        continue
    year_length = mir_trends["frac_new_dis_per_year"].shape[0]
    # print(year_length)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(list(range(year_length)), mir_trends["frac_new_dis_per_year"])
    
    slopes.append(slope)
    rs.append(r_value)
    ps.append(p_value)
    cepts.append(intercept)
#%%
#%%
#for a given mir/slope, plot the frac_new_dis_ler_year and also plot the fitted line

# mir = "hsa-mir-18a"
mir = most_causal_mirs["mir"].iloc[35]

index = most_causal_mirs["mir"].tolist().index(mir)
data = mir_to_mir_trends[mir][(mir_to_mir_trends[mir]["year"] >= 2012) & (mir_to_mir_trends[mir]['year'] <= 2020)].copy()

slope = slopes[index]
intercept = cepts[index]

year_length = data["frac_new_dis_per_year"].shape[0]

# Calculate the values of the fitted line
data['fitted_line'] = np.array(list(range(year_length))) * slope + intercept

# Plotting both the data and the fitted line
fig = px.line(data, x='year', y='frac_new_dis_per_year', title='Data and Fitted Line')
fig.add_scatter(x=data['year'], y=data['fitted_line'], mode='lines', name='Fitted Line')

# Show the figure
fig.show()

#%%

#%%
np.where(np.array(slopes) > 0)
#%%
sum(rs) / len(rs)
#%%
sum(ps) / len(ps)
#%%
# a = np.array(ps)
sig_idxs = np.where( np.array(ps) < .05)[0]


sig_slopes = np.array(slopes)[sig_idxs]

px.histogram(sig_slopes)
#%%
np.array(ps)[sig_idxs].mean()
#%%
sig_slopes

#%%
pleiotropic_mirs

#%%
#* clustering sthit

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X = pleiotropic_mirs[pleiotropic_mirs.columns[2:]].to_numpy()

pca = PCA(n_components=10)
X = StandardScaler().fit_transform(X)
pca.fit(X)

#%%
pcs = pca.transform(X)

color = "percent_cancer"
fig = px.scatter(x=pcs[:, 0], y=pcs[:, 1], hover_data=[pleiotropic_mirs["mir"]], color=pleiotropic_mirs[color], title=f"PCs, coloring {color}")
fig.show()
#%%
pca.explained_variance_ratio_

#%%
#import cosine similarity from torch
from scipy.spatial import distance

1 - distance.cosine(pcs[0], pcs[40])
# cosine_similarity(pcs[0], pcs[1])
#%%
mir_mat = np.empty((1291, 1291))

for i, mir in enumerate(pleiotropic_mirs["mir"]):
    for j, mir in enumerate(pleiotropic_mirs["mir"]):
        mir_mat[i][j] = 1 - distance.cosine(pcs[i], pcs[j])
#%%
mir_mat_df = pd.DataFrame(mir_mat, columns = pleiotropic_mirs["mir"], index = pleiotropic_mirs["mir"])

px.imshow(cluster_corr(mir_mat_df), color_continuous_midpoint=0, color_continuous_scale="RdBu")
#%%

mir_mat_metric = np.empty((1291, 1291))
for i, mir_i in tqdm(enumerate(mir_mat_df.columns)):
    for j, mir_j in enumerate(mir_mat_df.columns):
        mir_mat_metric[i][j] = pleiotropic_mirs[pleiotropic_mirs["mir"] == mir_i]["percent_cancer"].values[0] * pleiotropic_mirs[pleiotropic_mirs["mir"] == mir_j]["percent_cancer"].values[0]
#%%
mir_i = "hsa-mir-21"
pleiotropic_mirs[pleiotropic_mirs["mir"] == mir_i]["percent_cancer"].values[0]  * pleiotropic_mirs[pleiotropic_mirs["mir"] == "hsa-mir-21"]["percent_cancer"].values[0]
#%%
px.imshow(mir_mat_metric, color_continuous_midpoint=0, color_continuous_scale="RdBu")

#%%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import umap

reducer = umap.UMAP()

embedding = reducer.fit_transform(X)
embedding.shape

#%%

color = "percent_cancer"
fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], 
                 hover_data=[pleiotropic_mirs["mir"]], 
                 color=pleiotropic_mirs[color], 

                 title=f"UMAP of miRNA data")
fig.update_layout(coloraxis_colorbar_title="Percent Cancer",
                                   xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",)

fig.show()

#%%
pleiotropic_mirs[pleiotropic_mirs['mir'] == 'hsa-mir-1979']

#%%
pleiotropic_mirs


#%%
#* refactoring diseases:
import pickle
mesh_disease_to_doid = pickle.load(open("databases/mesh_diseases_to_doid.p", "rb"))
mesh_disease_to_doid

#find mesh diseases that have the same doid_ancestor
doid_ancestor_to_mesh_diseases = {}
for mesh_disease in mesh_disease_to_doid:
    doid_ancestor = mesh_disease_to_doid[mesh_disease]["doid_ancestor"]
    
    for anc in set(doid_ancestor):
        if anc not in doid_ancestor_to_mesh_diseases.keys():
            doid_ancestor_to_mesh_diseases[anc] = []
        
        doid_ancestor_to_mesh_diseases[anc].append(mesh_disease)

for doid_ancestor in doid_ancestor_to_mesh_diseases:
    if len(doid_ancestor_to_mesh_diseases[doid_ancestor]) > 1:
        print(doid_ancestor_to_mesh_diseases[doid_ancestor])
        print(doid_ancestor)
        print()

#%%
mesh_disease_to_doid["Diabetes Insipidus, Nephrogenic"]
        
#%%
#* LINEAR REGRESSIONS
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS, WLS

from sklearn.model_selection import train_test_split

def get_test_acc(X, y, test_size = 0.2, num_repeats = 10):
    
    scores = []
    for _ in range(num_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        scores.append(lr.score(X_test, y_test))

    return np.mean(scores), np.std(scores)

pleiotropic_mirs = pd.read_csv("databases/pleiotropic_mirs.csv")

start = 3

X = pleiotropic_mirs.iloc[:, start:].to_numpy()
y = pleiotropic_mirs['num_diseases'].to_numpy()

print(f"Test R2 on All Vars: {get_test_acc(X, y)}")

full_lr = LinearRegression().fit(X, y)
print(full_lr.score(X, y))

full_lr = OLS(y, X).fit()
print(pd.DataFrame(np.vstack([pleiotropic_mirs.iloc[:, start:].columns, full_lr.pvalues])))
full_lr.tvalues

full_lr.params
#%%
full_lr.params.shape
#%%
start = 4
no_paper_X = pleiotropic_mirs.iloc[:, start:].to_numpy()
y = pleiotropic_mirs['num_diseases'].to_numpy()

print(f"Test R2 on All Vars without num_papers: {get_test_acc(no_paper_X, y)}")

no_paper_lr = LinearRegression().fit(no_paper_X, y)

print(no_paper_lr.score(no_paper_X, y))

#%%
#*RESIDS

just_papers = pleiotropic_mirs.iloc[:, 3].to_numpy().reshape(-1, 1)
y = pleiotropic_mirs['num_diseases'].to_numpy()

print(f"Test R2 on just num_papers: {get_test_acc(just_papers, y)}")

just_papers_lr = LinearRegression().fit(just_papers, y)

print(just_papers_lr.score(just_papers, y))

paper_resids = just_papers_lr.predict(just_papers) - y

pleiotropic_mirs['paper_resids'] = paper_resids
#%%
just_papers_lr.coef_
#%%
pleiotropic_mirs['no_paper_lr_pred'] = no_paper_lr.predict(no_paper_X)
pleiotropic_mirs['full_lr_pred'] = full_lr.predict(X)

#%%
#*years resid
years = 2023- pleiotropic_mirs["miRbase_year"]
years[years == 2023] = 0
years = years.to_numpy()

pleiotropic_mirs['miRbase_year'] = years

#get resids for just years
just_years = pleiotropic_mirs.iloc[:, 9].to_numpy().reshape(-1, 1)
y = pleiotropic_mirs['num_diseases'].to_numpy()

print(f"Test R2 on just num_years: {get_test_acc(just_years, y)}")

just_years_lr = LinearRegression().fit(just_years, y)

print(just_years_lr.score(just_years, y))

years_resids = just_years_lr.predict(just_years) - y

pleiotropic_mirs['years_resids'] = years_resids

#%%
#*finding understudied mirnas

pleiotropic_mirs = pd.read_csv("databases/pleiotropic_mirs.csv")

full_resids = pleiotropic_mirs['full_lr_pred'] - pleiotropic_mirs['num_diseases']

full_resids = full_resids / np.std(full_resids)
pleiotropic_mirs['full_resids'] = full_resids

plot_pleiotropy(pleiotropic_mirs, "full_resids", "num_papers", "Full Residuals vs papers", hover_data=["mir"])
#%%
px.histogram(pleiotropic_mirs['full_resids'])
#%%
px.histogram(pleiotropic_mirs['num_papers'])
#%%

pleiotropic_mirs['num_papers'].median()

#%%
pleiotropic_mirs[pleiotropic_mirs['full_resids'] > 0]['full_resids'].mean()

#%%
frs = pleiotropic_mirs[(pleiotropic_mirs['full_resids'] > 1.25) & (pleiotropic_mirs['num_papers'] < 100)]

#%%
frs.to_csv("understudied_mirs.csv", index = False)

# %%

plot_pleiotropy(pleiotropic_mirs, "paper_resids", "num_papers", "Full Residuals vs papers", hover_data=["mir"])
# %%
pleiotropic_mirs[pleiotropic_mirs["mir"] == 'hsa-mir-451b']
# %%
pleiotropic_mirs[pleiotropic_mirs["mir"] == 'hsa-mir-451a']

#%%
pleiotropic_mirs["num_diseases"].mean()
# %%

#*non-specific mirs

#get all the mirnas that are causally implicated in 30 or more diseases

non_specific_mirs = pleiotropic_mirs[pleiotropic_mirs["num_diseases"] >= 64]

#%%
non_specific_mirs.drop(columns = ["diseases", "years_resids","paper_resids", "full_lr_pred", "num_diseases_thresh_2", "no_paper_lr_pred"], inplace = True)

#%%
non_specific_mirs
#%%

non_specific_mirs.to_csv("non_specific_mirs.csv", index = False)
# %%
#turn non_specific_mirs into a dataframe where each row represents a mir-disease causal link. the two columns are the mir name and the disease name and

# Initialize an empty list to collect rows for the new DataFrame
new_rows = []

# Iterate through each row in non_specific_mirs DataFrame
for index, row in non_specific_mirs.iterrows():
    mir_name = row['mir']  # Assuming the column for microRNA name is 'mir_name'
    diseases = row['diseases'].split("|")  # Assuming the column for diseases is a list or array stored under 'diseases'
    
    # Iterate through each disease for this miR
    for disease in diseases:
        new_row = {'mir': mir_name, 'disease': disease}
        new_rows.append(new_row)

# Create the new DataFrame
mir_disease_df = pd.DataFrame(new_rows)
#%%
mir_disease_df.to_csv("non-specific-mirs.csv", index = False)

# %%
mir_disease_df[mir_disease_df["disease"] == "Adenocarcinoma"]
# %%
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

# Read the CSV into a DataFrame (assuming columns are 'mir' and 'disease')
df = mir_disease_df

# Create an empty bipartite graph
B = nx.Graph()

# Add nodes with the node attribute "bipartite"
B.add_nodes_from(df['mir'].unique(), bipartite=0)
B.add_nodes_from(df['disease'].unique(), bipartite=1)

# Add edges only between nodes of opposite node sets
B.add_edges_from([(row['mir'], row['disease']) for idx, row in df.iterrows()])

# Create a projected graph
# Get the set of disease nodes
disease_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==1}

# Project the bipartite graph onto disease nodes
G = bipartite.projected_graph(B, disease_nodes)

# Now G is a graph connecting diseases that share a miRNA.
# You can write it to a new CSV if you'd like.
nx.write_edgelist(G, "disease_disease_projection.csv", delimiter=",", data=False)

# %%

pos = nx.spring_layout(G, seed=42)  # positions for all nodes
nx.draw(G, pos,
        with_labels=True,
        node_color='skyblue',
        node_size=2,
        font_size=2,
        font_color='black',
        edge_color='gray')

plt.title("Disease-Disease Projection")
plt.show()
# %%

lens = np.array([len(c.split()) for c in causal_pubmed['completion'].values.tolist()])
np.mean(lens), np.std(lens)
# %%
n = 80
causal_pubmed.loc[n:n+10]
# %%
causal_pubmed.loc[86].completion
# %%
#* agreement statistics
labels_rs = pd.read_csv("databases/labels/pubmed_labels_v2_RS.csv")
labels_sb = pd.read_csv("databases/labels/pubmed_labels_v2_sb.csv")
labels_mjm = pd.read_csv("databases/labels/human_annotated_papers_mjm.csv", index_col = 0)
labels_kw = pd.read_csv("databases/labels/pubmed_labels_v2_krw.csv")

labels_rs = labels_rs.sort_values(by = "TI")
labels_sb = labels_sb.sort_values(by = "TI")
labels_kw = labels_kw.sort_values(by = "TI")
labels_mjm = labels_mjm.sort_values(by = "TI")

def disagreement_between_two(df1, df2):
    num_wrong = 0
    for (i, row), l1, l2 in zip(df1.iterrows(), df1.is_mir_causal.values, df2.is_mir_causal.values):
        if l1.lower() != l2.lower():
            num_wrong += 1
    return num_wrong

print(f"Agreements between KW RS {100 - disagreement_between_two(labels_kw, labels_rs)}")
print(f"Agreements between SB RS {100 - disagreement_between_two(labels_sb, labels_rs)}")
print(f"Agreements between KW SB {100 - disagreement_between_two(labels_kw, labels_sb)}")

# %%
from create_miraidd.lm_causal_utils import load_pubmed
pubmed = load_pubmed()

pubmed_test_v2 = pubmed.sample(n = 100, random_state = 1).sort_values(by = "TI")

tv2 = causal_pubmed[causal_pubmed["mir"].isin(pubmed_test_v2["mir"].values) & causal_pubmed["PMID"].isin(pubmed_test_v2["PMID"].values) & causal_pubmed["AB"].isin(pubmed_test_v2["AB"].values) & causal_pubmed["TI"].isin(pubmed_test_v2["TI"].values) ]
tv2.sort_values(by = "TI")
tv2 = causal_pubmed[causal_pubmed.index.isin(pubmed_test_v2.index)] 
tv2.rename(columns = {"causal": "is_mir_causal"}, inplace = True)

labels_gpt = tv2.sort_values(by = "TI")

#relabel the 1s to yes and 0s to no in labels_gpt
labels_gpt["is_mir_causal"] = ["yes" if l == 1 else "no" for l in labels_gpt["is_mir_causal"].values]

#%%
# Define the labels
#human 1, 2, 3, gpt
labels = [labels_rs, labels_sb, labels_kw, labels_gpt]

# Create an empty agreement matrix
agreement_matrix = np.zeros((4, 4))

# Calculate pairwise agreement between each set of labels
for i in range(4):
    for j in range(4):
        num_agreements = 0
        num_disagreements = 0
        for l1, l2 in zip(labels[i]['is_mir_causal'], labels[j]['is_mir_causal']):
            if l1.lower() == l2.lower():
                num_agreements += 1
            else:
                num_disagreements += 1
        agreement_matrix[i, j] = num_agreements / (num_agreements + num_disagreements)

# Print the agreement matrix
print(agreement_matrix)

# %%
#* comparing hmdd and pubmed
causal_pubmed = pd.read_csv("databases/miraidd.csv")

HMDD_db_path = 'databases/external_databases/HMDD3_causal_info.csv'
HMDD_db = pd.read_csv(HMDD_db_path, header = None, encoding= 'unicode_escape')

HMDD_db = HMDD_db[[1, 2, 3, 4, 5]].iloc[1:] #gets rid of header row and category column
HMDD_db.columns = ['mir', 'disease', 'mesh_name', 'PMID', 'causality']
#%%
#change datatype of column
HMDD_db['PMID'] = pd.to_numeric(HMDD_db['PMID'], errors='coerce')
causal_pubmed['PMID'] = pd.to_numeric(causal_pubmed['PMID'], errors='coerce')
#%%
HMDD_db
#%%
hmdd_miraidd_intersection = pd.merge(HMDD_db, causal_pubmed, on = ["PMID", "mir"], how = "inner")
#%%
hmdd_miraidd_intersection[hmdd_miraidd_intersection.mir == "hsa-mir-15a"]
# %%
num_agree = 0
for i, row in hmdd_miraidd_intersection.iterrows():
    if ((row["causality"] == "no" and row["causal"] == 0)) or ((row["causality"] == "yes" and row["causal"] == 1)):
        num_agree += 1
    else:
        print(row["AB"])
        print(row['completion'])
        print(row["causality"])
        print(row["causal"])
        print()
num_agree / len(hmdd_miraidd_intersection)
#%%
len(hmdd_miraidd_intersection)

# %%
disagreements = hmdd_miraidd_intersection[(hmdd_miraidd_intersection["causality"] == "no") & (hmdd_miraidd_intersection["causal"] == 1) | (hmdd_miraidd_intersection["causality"] == "yes") & (hmdd_miraidd_intersection["causal"] == 0)]
causal_pubmed
# %%
disagreements[['mir', 'disease', 'PMID', 'AB', 'TI', 'causality', 'causal', 'completion']].rename(columns={'causality': 'HMDD label', 'causal': 'miRAIDD label'}).to_csv("hmdd_miraidd_disagreements.csv", index = False)

#%%
disagreements = pd.read_csv("hmdd_miraidd_disagreements.csv")

miraiddno  = disagreements[disagreements['miRAIDD label'] == 0].iloc[:50]
miraiddyes = disagreements[disagreements['miRAIDD label'] == 1].iloc[:50]

filtered_disagree = pd.concat([miraiddno, miraiddyes])
filtered_disagree.to_csv("hmdd_miraidd_disagreements_samples.csv", index = False)
# %%

disagreements[['mir', 'disease', 'PMID',  'AB', 'TI', 'causality', 'causal', 'completion']]

# %%
pleiotropic_mirs_with_hsa_let_7 = [mir for mir in pleiotropic_mirs['mir'] if 'hsa-let-7' in mir]

# %%
pleiotropic_mirs[pleiotropic_mirs['mir'].str.contains('hsa-let-7')]

# %%
#* fixing mirmine 
pleiotropic_mirs[pleiotropic_mirs['mir'] == 'hsa-mir-21']['diseases'].values[0].split("|")

# %%

#* threshold

mir_to_disease2 = create_mir_to_diseases(disease_count = 2)
# %%
mir_to_disease1 = create_mir_to_diseases(disease_count = 1)

# %%
percent_change = []
for mir in mir_to_disease2:
    if mir in mir_to_disease1:
        pleiotropy2 = len(mir_to_disease2[mir])
        pleiotropy1 = len(mir_to_disease1[mir])
        percent_change.append((pleiotropy2 - pleiotropy1) / pleiotropy1 if pleiotropy1 != 0 else 0)


# %%
px.histogram(percent_change)
# %%
sum(percent_change) / len(percent_change)
# %%
