import pandas as pd
from tqdm import tqdm
import pickle

import requests
from urllib.parse import quote
from selenium import webdriver
import requests
from time import sleep
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options

import pandas as pd
from tqdm import tqdm
from datetime import date
import pickle
import numpy as np

from pleiotropic_factors import create_mir_to_diseases

def create_driver():
    options = Options()
    # options.headless = True
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument('--disable-gpu')
    options.add_argument("--no-sandbox")

    chrome_prefs = {}
    options.experimental_options["prefs"] = chrome_prefs
    chrome_prefs["profile.default_content_settings"] = {"images": 2}

    driver = webdriver.Chrome(executable_path='./chromedriver', options = options)
    print("driver successfully created")
    return driver

driver = create_driver()
all_diseases = []
for offset in range(0, 7000, 1000):
    url = f"https://id.nlm.nih.gov/mesh/query?query=PREFIX%20rdf%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F1999%2F02%2F22-rdf-syntax-ns%23%3E%0D%0APREFIX%20rdfs%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX%20xsd%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2001%2FXMLSchema%23%3E%0D%0APREFIX%20owl%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2002%2F07%2Fowl%23%3E%0D%0APREFIX%20meshv%3A%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%2Fvocab%23%3E%0D%0APREFIX%20mesh%3A%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%2F%3E%0D%0APREFIX%20mesh2023%3A%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%2F2023%2F%3E%0D%0APREFIX%20mesh2022%3A%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%2F2022%2F%3E%0D%0APREFIX%20mesh2021%3A%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%2F2021%2F%3E%0D%0A%0D%0ASELECT%20distinct%20%3Fd%20%3FdLabel%20%0D%0AFROM%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%3E%0D%0AWHERE%20%7B%0D%0A%20%20%3Fd%20meshv%3AallowableQualifier%20%3Fq%20.%0D%0A%20%20%3Fq%20rdfs%3Alabel%20%27pathology%27%40en%20.%20%0D%0A%20%20%3Fd%20rdfs%3Alabel%20%3FdLabel%20.%20%0D%0A%7D%20%0D%0AORDER%20BY%20%3FdLabel%20%0D%0A&limit=1000&inference=false&offset={offset}"

    driver.get(url)
    driver.implicitly_wait(10)

    els = driver.find_elements_by_xpath("//*[@id='lodestar-results-table']/tr[*]/td[2]")

    diseases = [el.get_attribute("textContent") for el in els]
    
    all_diseases += diseases

#flatten all_diseases
print(all_diseases)
pickle.dump(all_diseases, open('mesh_diseases.p', 'wb'))

mesh_diseases = pickle.load(open("mesh_diseases.p", "rb"))

class DOID_terms():
    def __init__(self):
        self.doid_terms = []
    
    def is_term_in_doid(self, query):
        if query in self.doid_terms:
            return True
        
        api_url = "http://www.ebi.ac.uk/ols/api/search?q=" + query + "&ontology=doid&exact=True"
        response = requests.get(api_url)
            
        for term in response.json()['response']['docs']:
            if term['is_defining_ontology'] == True:
                self.doid_terms.append(query)
                return True
        
        return False
    
#FILTERING MESH DISEASES
url = "https://id.nlm.nih.gov/mesh/sparql"

doid_t = DOID_terms()
mesh_diseases_to_doid = {}

count = 0

for d in tqdm(mesh_diseases):
    print(d)
    if doid_t.is_term_in_doid(d):
        print("exact match")
        mesh_diseases_to_doid[d] = {
            "in_doid_exact": True,
            "doid_ancestor": ""
        }
    else:
        #get mesh term ancestors
        query = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
        PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
        PREFIX mesh2023: <http://id.nlm.nih.gov/mesh/2023/>
        PREFIX mesh2022: <http://id.nlm.nih.gov/mesh/2022/>
        PREFIX mesh2021: <http://id.nlm.nih.gov/mesh/2021/>

        SELECT ?d ?dName 
        FROM <http://id.nlm.nih.gov/mesh>
        WHERE {
        ?d a meshv:Descriptor .
        ?d meshv:concept ?c .
        ?d rdfs:label ?dName .
        FILTER(str(?dName) = '""" + d + """')
        } 
        ORDER BY ?d 
        """
        
        response = requests.get(url, params = {"query" : query, "format": "JSON", "inference": "true"})
        
        if response.status_code == 200:
            ret = response.json()

            assert len(ret["results"]["bindings"]) != 0
            
            mesh_url = [r['d']['value'] for r in ret["results"]["bindings"]][0]
            mesh_id = mesh_url.split("/")[-1]
            print(mesh_id)
            
            query = """
            PREFIX mesh: <http://id.nlm.nih.gov/mesh/> 
            PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#> 
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 

            SELECT ?treeNum ?ancestorTreeNum ?ancestor ?alabel
            FROM <http://id.nlm.nih.gov/mesh>
            WHERE {
            mesh:""" + str(mesh_id) + """ meshv:treeNumber ?treeNum .
            ?treeNum meshv:parentTreeNumber+ ?ancestorTreeNum .
            ?ancestor meshv:treeNumber ?ancestorTreeNum .
            ?ancestor rdfs:label ?alabel
            }
            ORDER BY ?treeNum ?ancestorTreeNum
            """

            response = requests.get(url, params = {"query" : query, "format": "JSON", "inference": "true"})
            
            if response.status_code == 200:
                ret = response.json()
                # print(ret)
                ancestors = [r['alabel']['value'] for r in ret["results"]["bindings"]]
                
                print(ancestors)
                doid_ancestors = [a for a in ancestors if doid_t.is_term_in_doid(a)]
                
                if len(doid_ancestors) > 0:
                    print("ancestor match")
                    mesh_diseases_to_doid[d] = {
                        "in_doid_exact": False,
                        "doid_ancestor": doid_ancestors,
                    }                    
            else:
                print("mesh term has no ancestors")
                mesh_diseases_to_doid[d] = {
                "in_doid_exact": False,
                "doid_ancestor": "",
                    }
        else:
            print("mesh term has no mesh id")
            mesh_diseases_to_doid[d] = {
            "in_doid_exact": False,
            "doid_ancestor": "",
                }
        
        if d not in mesh_diseases_to_doid.keys():
            print("no match")
            mesh_diseases_to_doid[d] = {
            "in_doid_exact": False,
            "doid_ancestor": "",
                }
        
        print()
        count += 1
        
        if count > 100:
            print(mesh_diseases_to_doid)
            count = 0

pickle.dump(mesh_diseases_to_doid, open('mesh_diseases_to_doid.p', 'wb'))

#nohup python mesh_n_doid.py &> mesh_n_doid.out &
