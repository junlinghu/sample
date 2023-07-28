
import pandas as pd

def load_csv(file_name):
    df = pd.read_csv(file_name, sep='\t')
    total=[]
    for i, row in df.iterrows():
        x1=int(row['x1'])
        x2=int(row['x2'])
        plus=x1+x2
        total.append(plus)
        print(i, x1, x2, plus)
    df['total'] = total
    df.to_csv("test_new.csv", sep='\t', index=False)


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
model.to(device)

import numpy as np

def generate_embedding(file_name, npy_file):
    df = pd.read_csv(file_name, sep='\t')
    sentences = [str(row['title']) + '. ' + row['text'] for _, row in df.iterrows()]
    embeddings = model.encode(sentences)
    np.save('data/' + npy_file, embeddings)

if __name__ == '__main__':
	#file_name = "test_number.csv"
	#load_csv(file_name)
    # file_name = "test_text.csv"
    # npy_file = "test.npy"
    # generate_embedding(file_name,npy_file)
    wget https://www.dropbox.com/s/648ddugqtimjgxp/churn.csv
