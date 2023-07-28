
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
import numpy as np

def generate_embedding(file_name, npy_file):
    model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
    model.to(device)
    
    df = pd.read_csv(file_name, sep='\t')
    sentences = [str(row['title']) + '. ' + row['text'] for _, row in df.iterrows()]
    embeddings = model.encode(sentences)
    np.save('data/' + npy_file, embeddings)

import subprocess
def download_file(url):
    command = "wget {}".format(url)
    subprocess.run(command, shell=True)
    
from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.get_layer = nn.Sequential(
                            nn.Linear(18, 20),
                            nn.ReLU(),
                            nn.Linear(20, 10),
                            nn.ReLU(),
                            nn.Linear(10, 2),)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.get_layers(x)
        return logits

def classify():
    model = NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 18, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

if __name__ == '__main__':
	#file_name = "test_number.csv"
	#load_csv(file_name)
    # file_name = "test_text.csv"
    # npy_file = "test.npy"
    # generate_embedding(file_name,npy_file)
    
    # url="https://www.dropbox.com/s/648ddugqtimjgxp/churn.csv"
    # download_file(url)
    
    classify()
    
    
