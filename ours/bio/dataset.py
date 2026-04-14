import torch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import math
from peptide_encoding import feature_generator

def read_protein_sequences_from_fasta(file_path):
    sequences = []
    labels = []
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
                if 'pos' in line:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences, labels

def load_encoding_from_txt(file_path):
    sequences, labels = read_protein_sequences_from_fasta(file_path)
    encoded_sequences = feature_generator(file_path)
    return encoded_sequences, labels

def load_features_from_txt(feature_file_path):
    features = np.loadtxt(feature_file_path)
    return features

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, features, labels):
        self.input_ids = input_ids
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
