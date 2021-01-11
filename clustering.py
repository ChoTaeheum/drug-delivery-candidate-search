#%% build environment
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:59:55 2020

@author: modn
"""

####
import os
current_path = '//192.168.0.50/projects/cth_Polarity/datasets/drugbank_20'
os.chdir(current_path)
####
import pickle
import numpy as np
import pandas as pd
import sklearn as sk
import scipy.stats

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder



dir_path = './'
file_list = os.listdir(dir_path)

result = pd.DataFrame({'km0':[], 'km1':[], 'sc0':[], 'sc1':[], 'ac':[]})

amph_coords = {}
for file in file_list:
    amph_coords[file[:-4]] = np.array(pd.read_csv(file))
    
# 하나의 sample에 대해서 test
sample = amph_coords['NPC208650_hydrophilic_coord']


#%% constants
A1 = 0.8
A2 = 0.8

#%% class definition
class ClusteringEnsemble:
    def __init__(self):
        self.a1 = 0.8
        self.a2 = 0.8        
        self.clt_result = pd.DataFrame({'km0':[], 'km1':[], 'sc0':[], 'sc1':[], 'ac':[]})

    def transformation(self, sample):
        km0 = KMeans(n_clusters=2, random_state=0).fit(sample)
        self.clt_result['km0'] = km0.labels_
        
        km1 = KMeans(n_clusters=2, random_state=111).fit(sample)
        self.clt_result['km1'] = km1.labels_
        
        sc0 = SpectralClustering(n_clusters=2,  random_state=0).fit(sample)
        self.clt_result['sc0'] = sc0.labels_
        
        sc1 = SpectralClustering(n_clusters=2,  random_state=111).fit(sample)
        self.clt_result['sc1'] = sc1.labels_
        
        ac = AgglomerativeClustering(n_clusters=2).fit(sample)
        self.clt_result['ac'] = ac.labels_
        
        enc = OneHotEncoder()
        enc.fit(np.array(self.clt_result['km0']).reshape(-1, 1))
        
        km0_onehot = enc.transform(np.array(self.clt_result['km0']).reshape(-1, 1)).toarray()
        km1_onehot = enc.transform(np.array(self.clt_result['km1']).reshape(-1, 1)).toarray()
        sc0_onehot = enc.transform(np.array(self.clt_result['sc0']).reshape(-1, 1)).toarray()
        sc1_onehot = enc.transform(np.array(self.clt_result['sc1']).reshape(-1, 1)).toarray()
        ac_onehot = enc.transform(np.array(self.clt_result['ac']).reshape(-1, 1)).toarray()
        
        onehot_cst = pd.DataFrame(np.hstack((km0_onehot, km1_onehot, sc0_onehot, sc1_onehot, ac_onehot)),
                                     columns=('km0_0', 'km0_1',
                                              'km1_0', 'km1_1',
                                              'sc0_0', 'sc0_1',
                                              'sc1_0', 'sc1_1',
                                              'ac_0', 'ac_1'))
        
        return onehot_cst
    
    
    def consensus_function(self, onehot_cst): 
        while len(onehot_cst.columns) > 2:    # replace constant!
            # calculate cluster similarity
            cluster_sim = onehot_cst.corr()    # Pearson's R, cluster similarity
            
            # cluster similarity matrix -> upper triangular matrix
            mask = (np.triu(np.ones(len(onehot_cst.columns))) - np.eye(len(onehot_cst.columns))).astype(np.bool)
            cluster_sim = cluster_sim.where(mask)
            cluster_sim = cluster_sim.stack().reset_index()
            cluster_sim.columns = ['cluster1', 'cluster2', 'similarity']
            
            # extract argmax cluster similarity
            if cluster_sim['similarity'].max() > A1:
                maxval_pair = cluster_sim.iloc[cluster_sim['similarity'].idxmax()]
            else:
                return onehot_cst
            
            # update cluster matrix
            new_cst = onehot_cst[maxval_pair['cluster1']] + onehot_cst[maxval_pair['cluster2']]
            onehot_cst[maxval_pair['cluster1'] + '+' + maxval_pair['cluster2']] = new_cst
            onehot_cst = onehot_cst.drop(columns=[maxval_pair['cluster1'], maxval_pair['cluster2']])
    
        return onehot_cst
    
    
    def membership_similarity(self, onehot_cst):
        membership_sim = onehot_cst / max(onehot_cst.max())
        return membership_sim
        
        
    def elimination(self, onehot_cst):
        membership_sim = self.membership_similarity(onehot_cst)
        cluster_qual = membership_sim[membership_sim!=0].mean()
        cluster_qual = cluster_qual.sort_values(ascending=False)[:3]
        membership_sim = membership_sim[list(cluster_qual.index)]
        return membership_sim
        
    
    def minimum_effect(self, membership_sim):
        cluster_var = membership_sim[membership_sim>A2].var()
        certain_mask = membership_sim >= A2
        
        determined_cst = certain_mask
        uncertain_list = membership_sim.max(axis=1) < A2
        uncertain_list = list(uncertain_list[uncertain_list==True].index)    
        for i in uncertain_list:
            candidate_test = certain_mask
            candidate_test.iloc[i, :] = True
            candidate_var = membership_sim[candidate_test].var()
            elected = abs(cluster_var - candidate_var).idxmin()
            determined_cst[elected][i] = True
        
        membership_sim = membership_sim[determined_cst].fillna(0)
        return membership_sim
        
    
    def enforce_clustering(self, membership_sim):
        membership_sim.columns = list(range(len(membership_sim.columns)))    # 컬럼이름 숫자로 바꾸기
        
        if (membership_sim.max(axis=1) > A2).all() == False:    # when all objects are not certain
            membership_sim = self.minimum_effect(membership_sim)
            
        last_cluster = membership_sim.idxmax(axis=1)
        
        return last_cluster

    def ensemble(self, sample):
        onehot_cst = self.transformation(sample)
        onehot_cst = self.consensus_function(onehot_cst)
        membership_sim = self.elimination(onehot_cst)
        last_cluster = self.enforce_clustering(membership_sim)
        
        return last_cluster
#%%    

import pickle
import numpy as np
import pandas as pd
import sklearn as sk
import scipy.stats

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv("Y:/tmp/db33114.csv")   
 
clustering_ensemble = ClusteringEnsemble()
last_cluster = clustering_ensemble.ensemble(data)

#%%
clustering_ensemble.transformation(data)

#%%수동으로 해보기

clt_result = pd.DataFrame({'km0':[], 'km1':[], 'sc0':[], 'sc1':[], 'ac':[]})

    
km0 = KMeans(n_clusters=2, random_state=0).fit(data)
clt_result['km0'] = km0.labels_
        
km1 = KMeans(n_clusters=2, random_state=111).fit(data)
clt_result['km1'] = km1.labels_
        
sc0 = SpectralClustering(n_clusters=2,  random_state=0, eigen_solver="arpack").fit(data)
clt_result['sc0'] = sc0.labels_
        
sc1 = SpectralClustering(n_clusters=2,  random_state=111).fit(data)
clt_result['sc1'] = sc1.labels_
        
ac = AgglomerativeClustering(n_clusters=2).fit(data)
clt_result['ac'] = ac.labels_
        
enc = OneHotEncoder()
enc.fit(np.array(self.clt_result['km0']).reshape(-1, 1))
        
km0_onehot = enc.transform(np.array(self.clt_result['km0']).reshape(-1, 1)).toarray()
km1_onehot = enc.transform(np.array(self.clt_result['km1']).reshape(-1, 1)).toarray()
sc0_onehot = enc.transform(np.array(self.clt_result['sc0']).reshape(-1, 1)).toarray()
sc1_onehot = enc.transform(np.array(self.clt_result['sc1']).reshape(-1, 1)).toarray()
ac_onehot = enc.transform(np.array(self.clt_result['ac']).reshape(-1, 1)).toarray()
        
onehot_cst = pd.DataFrame(np.hstack((km0_onehot, km1_onehot, sc0_onehot, sc1_onehot, ac_onehot)),
                          columns=('km0_0', 'km0_1',
                                   'km1_0', 'km1_1',
                                   'sc0_0', 'sc0_1',
                                   'sc1_0', 'sc1_1',
                                   'ac_0', 'ac_1'))
        



#%% excution
if __name__ == '__main__':
    clustering_ensemble = ClusteringEnsemble()
    last_cluster = clustering_ensemble.ensemble(sample)

        
