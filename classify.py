
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


if __name__ == '__main__':
	file_name = "test_number.csv"
	load_csv(file_name)
