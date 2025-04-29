import numpy as np
import torch
import torch.nn as nn
from data_deal import convertToGraph, convertToGraph_add_smiles
from torch.utils.data import Dataset
import os
import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, Descriptors, BRICS
import pandas as pd

class ToxicDataset(Dataset):
    def __init__(self, x, y, z):
        super(ToxicDataset, self).__init__()
        self.x = x
        self.y = y
        self.z = z
    def __len__(self):
        return len(self.z)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.float32(self.x[idx])),
            torch.from_numpy(np.float32(self.y[idx])),
            torch.from_numpy(np.float32(self.z[idx]))
        )

class newToxicDataset(Dataset):
    def __init__(self, x, y, z):
        super(newToxicDataset, self).__init__()
        self.x = x
        self.y = y
        self.z = z
    def __len__(self):
        return len(self.z)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.float32(self.x[idx])),
            torch.from_numpy(np.float32(self.y[idx])),
            torch.from_numpy(self.z[idx])
        )

def loadInputs_train(args):
    adj = np.load(args.dataset + '/adj/' + 'train.npy')
    features = np.load(args.dataset + '/features/' + 'train.npy')
    y_train = np.load(args.dataset + '/train_target.npy')
    y_train = y_train[:, None]
    return features, adj, y_train

def loadInputs_feature_smiles(args, names):
    adj = np.load(args.dataset + '/adj/' + names + '.npy')
    features = np.load(args.dataset + '/features/' + names + '.npy')
    y = np.load(args.dataset + '/' + names + '_target.npy')
    y = y[:, None]
    smiles = np.load(args.dataset + '/' + names + '_smiles.npy')
    return features, adj, y, smiles

def loadInputs_val(args):
    adj = np.load(args.dataset + '/adj/' + 'val.npy')
    features = np.load(args.dataset + '/features/' + 'val.npy')
    y_val = np.load(args.dataset + '/val_target.npy')
    y_val = y_val[:, None]
    return features, adj, y_val

def loadInputs_test(args):
    adj = np.load(args.dataset + '/adj/' + 'test.npy')
    features = np.load(args.dataset + '/features/' + 'test.npy')
    y_test = np.load(args.dataset + '/test_target.npy')
    y_test = y_test[:, None]
    return features, adj, y_test

def ZW6(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=256).ToBitString()

def get_mol(smile):
    """Parses a list of SMILES or a single SMILES string into RDKit mol objects."""
    try:
        N = len(smile)
        nump = []
        for i in range(N):
            mol = AllChem.MolFromSmiles(smile[i])
            if mol is None:
                print(f"    [Warning] RDKit failed on SMILES: {smile[i]}")
            nump.append(mol)
        return nump
    except TypeError:
        nump = []
        mol = AllChem.MolFromSmiles(smile)
        if mol is None:
            print(f"    [Warning] RDKit failed on SMILES: {smile}")
        nump.append(mol)
        return nump

def get_morgan_feature(smile):
    """Generates a Morgan fingerprint (radius=3, 256 bits) for each SMILES."""
    mol = get_mol(smile)
    data = []
    for i in range(len(mol)):
        try:
            if mol[i] is not None:
                data.append([smile[i], ZW6(mol[i])])
            else:
                print(f"    [Warning] Could not generate Morgan FP for invalid SMILES: {smile[i]}")
        except Exception as e:
            print("    [Exception in get_morgan_feature]:", e)
            continue
    jak_feature = pd.DataFrame(data, columns=['smiles', 'ZW6'])
    num_frame6 = []
    for i in range(len(jak_feature['ZW6'])):
        num_frame6.append([x for x in jak_feature['ZW6'][i]])
    jak_zw6 = pd.DataFrame(num_frame6, dtype=np.float)
    return jak_zw6

def get_rdkit_descriptors(mol):
    """
    Returns 5 chosen classical descriptors for the given RDKit Mol.
    You can add or remove as needed.
    """
    if mol is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
    ]

def get_brics_vector(mol):
    """
    Returns BRICS fragment features
    """
    if mol is None:
        return [0.0, 0.0]
    frags = BRICS.BRICSDecompose(mol)
    if not frags:
        return [0.0, 0.0]
    frag_list = sorted(list(frags))
    return [
        float(len(frag_list)),
        float(sum(len(f) for f in frag_list)),
    ]

def load_data(args):

    train_path = os.path.join(args.dataset, 'train.csv')
    val_path   = os.path.join(args.dataset, 'val.csv')
    test_path  = os.path.join(args.dataset, 'test.csv')
    if not os.path.isfile(train_path):
        print(f"[Error] {train_path} does not exist!")
        return
    if not os.path.isfile(val_path):
        print(f"[Error] {val_path} does not exist!")
        return
    if not os.path.isfile(test_path):
        print(f"[Error] {test_path} does not exist!")
        return
    with open(train_path, 'r') as f:
        smiles_train = f.readlines()
    with open(val_path, 'r') as f:
        smiles_val = f.readlines()
    with open(test_path, 'r') as f:
        smiles_test = f.readlines()
    adj_dir = os.path.join(args.dataset, 'adj')
    feat_dir = os.path.join(args.dataset, 'features')
    if not os.path.exists(adj_dir):
        os.mkdir(adj_dir)
    if not os.path.exists(feat_dir):
        os.mkdir(feat_dir)
    train_adj, train_features, train_target, train_smiles = convertToGraph_add_smiles(smiles_train, 1)
    if len(train_adj) == 0:
        print(f"[Warning] No valid molecules parsed from TRAIN in {args.dataset}!")
    val_adj, val_features, val_target, val_smiles = convertToGraph_add_smiles(smiles_val, 1)
    if len(val_adj) == 0:
        print(f"[Warning] No valid molecules parsed from VAL in {args.dataset}!")
    test_adj, test_features, test_target, test_smiles = convertToGraph_add_smiles(smiles_test, 1)
    if len(test_adj) == 0:
        print(f"[Warning] No valid molecules parsed from TEST in {args.dataset}!")
    def process_desc_frag(lines):
        desc_list = []
        frag_list = []
        for line in lines:
            if line.startswith('smiles'):
                continue
            row = line.strip().split(',')
            if len(row) < 2:
                continue
            smi = row[0]
            mol = Chem.MolFromSmiles(smi)
            desc_list.append(get_rdkit_descriptors(mol))
            frag_list.append(get_brics_vector(mol))
        return np.array(desc_list, dtype=float), np.array(frag_list, dtype=float)
    train_desc, train_frag = process_desc_frag(smiles_train)
    val_desc,   val_frag   = process_desc_frag(smiles_val)
    test_desc,  test_frag  = process_desc_frag(smiles_test)
    try:
        np.save(os.path.join(adj_dir, 'train.npy'), train_adj)
        np.save(os.path.join(feat_dir, 'train.npy'), train_features)
        np.save(os.path.join(args.dataset, 'train_target.npy'), train_target)
        np.save(os.path.join(args.dataset, 'train_smiles.npy'), train_smiles)
        np.save(os.path.join(adj_dir, 'val.npy'), val_adj)
        np.save(os.path.join(feat_dir, 'val.npy'), val_features)
        np.save(os.path.join(args.dataset, 'val_target.npy'), val_target)
        np.save(os.path.join(args.dataset, 'val_smiles.npy'), val_smiles)
        np.save(os.path.join(adj_dir, 'test.npy'), test_adj)
        np.save(os.path.join(feat_dir, 'test.npy'), test_features)
        np.save(os.path.join(args.dataset, 'test_target.npy'), test_target)
        np.save(os.path.join(args.dataset, 'test_smiles.npy'), test_smiles)
        np.save(os.path.join(args.dataset, 'train_desc.npy'), train_desc)
        np.save(os.path.join(args.dataset, 'val_desc.npy'),   val_desc)
        np.save(os.path.join(args.dataset, 'test_desc.npy'),  test_desc)
        np.save(os.path.join(args.dataset, 'train_frag.npy'), train_frag)
        np.save(os.path.join(args.dataset, 'val_frag.npy'),   val_frag)
        np.save(os.path.join(args.dataset, 'test_frag.npy'),  test_frag)

    except Exception as e:
        print("[Error] Could not save featurized .npy files:", e)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    load_data(args)