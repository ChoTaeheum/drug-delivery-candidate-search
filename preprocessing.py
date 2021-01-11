# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:39:46 2020

@author: modn
"""

####
import os
current_path = '//192.168.0.50/projects'
os.chdir(current_path)
####
#%%
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import mdtraj
import numpy as np
from dgllife.utils import mol_to_bigraph
import pickle
import random
from rdkit.Chem import rdDepictor
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch
import time
import pickle
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA
import pandas as pd

#%%
dataset = pd.read_csv('cth_NPASS/datasets/ginsenoside_smiles.csv')
smiles_input_dic = {}
for i, row in dataset.iterrows():
    smiles_input_dic[row['name']] = row['smiles']

ERR_NUMS = []
    
#%%
class SmilesToInput:
    def __init__(self, smiles_input_dic, coords = {}, adj = {}, feat = {}, smiles = {}, pps = {}):
        self.smiles_input = smiles_input_dic
        self.coords_dic = coords
        self.adj_dic = adj
        self.feat_dic = feat
        self.smiles_dic = smiles
        self.pps = pps

    def make_dic(self):     
        def featurize_bonds(mol):
            feats = []
            bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
            for bond in mol.GetBonds():
                btype = bond_types.index(bond.GetBondType())
                # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
                feats.extend([btype, btype])
            return {'type': torch.tensor(feats).reshape(-1, 1).float()}
        
        for idx, (key, value) in enumerate(self.smiles_input.items()):   # 각 molecule에 대해
            try:
                mol = Chem.MolFromSmiles(value)                 ## Mol to Smiles
                mol = Chem.RemoveHs(mol)
                mol = Chem.AddHs(mol)

                rdDepictor.Compute2DCoords(mol,canonOrient=False)
                AllChem.EmbedMultipleConfs(mol, 5)    # 원하는 수만큼 conformer pool 생성
                confs = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=1000)
                engys = [c[1] for c in confs]
                min_e_idx = engys.index(min(engys))
                conf = mol.GetConformer(min_e_idx)

                x, y, z = [], [], []
                for k in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(k)
                    x.append(pos.x); y.append(pos.y); z.append(pos.z)
                x = np.array(x); y = np.array(y); z = np.array(z)
                codi = np.vstack((x, y, z)).T
                self.coords_dic[key] = codi                     # 좌표 저장


                grp = mol_to_bigraph(mol, edge_featurizer=featurize_bonds, canonical_atom_order=False)
                grp_mat = grp.adjacency_matrix()
                grp_mat = np.array(grp_mat.to_dense())

                num_atoms = mol.GetNumAtoms()
                for i in range(num_atoms):
                    for j in range(num_atoms):
                        if grp.edge_id(i,j, return_array=1).size()[0] != 0:
                            grp_mat[j,i] += int(grp.edata['type'][int(grp.edge_id(i,j, return_array=1))])
                self.adj_dic[key] = grp_mat                      # 그래프 저장

                atoms = mol.GetAtoms()
                atomicnums = []
                for j, atom in enumerate(atoms):
                    atomicnums.append(atom.GetAtomicNum())
                self.feat_dic[key] = np.array(atomicnums)        # 원자번호 저장

                self.smiles_dic[key] = value                    # smiles 저장

                tpsa = Descriptors.TPSA(mol)
                radii = rdFreeSASA.classifyAtoms(mol)
                sasa = rdFreeSASA.CalcSASA(mol, radii)
                self.pps[key] = tpsa / sasa                     # pps 저장
                
            except:
                print(idx, key)
                ERR_NUMS.append(key)
            
        return self.adj_dic, self.feat_dic, self.coords_dic, self.smiles_dic, self.pps
    
#%%
smiles_to_input = SmilesToInput(smiles_input_dic)
a, b, c, d, e = smiles_to_input.make_dic()

#%%
with open('ginsenoside_dic.txt', 'wb') as f:
    pickle.dump((a, b, c, d, e), f)
