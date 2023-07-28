
import pandas as pd

def load_csv(file_name):
    df = pd.read_csv(file_name, sep='\t')
    
    print("Processing ...")
    for number in df['x1']:
        print("")
    df['summary'] = summarized_texts

    df.to_csv("test_new.csv", sep='\t', index=False)


if __name__ == '__main__':
	file_name = "test.csv"
	load_csv(file_name)
