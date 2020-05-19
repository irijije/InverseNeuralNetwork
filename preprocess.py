import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BASE_PATH = "dataset/"
FILE_PATH = ""

def drop_columns():
    chunks = pd.read_csv(BASE_PATH+FILE_PATH+"Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", chunksize=1000000)
    df = []
    for chunk in chunks:
        print("chunk processing")
        chunk = chunk.drop(['Flow ID', 'Src IP', 'Src Port', 'Dst IP'], axis=1)
        df.append(chunk)
    pd.concat(df).to_csv(BASE_PATH+FILE_PATH+"Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", index=False)

def make_dataset():
    all_data = []
    for f in glob.glob(BASE_PATH+FILE_PATH+"*.csv"):
        data = pd.read_csv(f, index_col=None, header=0)
        all_data.append(data)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    data = data.drop(['Timestamp'], axis=1)
    data = data[~data['Dst Port'].str.contains("Dst", na=False)]
    data_train, data_test = train_test_split(data, test_size=0.2)
    data.to_csv(BASE_PATH+"CICIDS2018_all.csv", index=False)
    pd.DataFrame(data_train).to_csv(BASE_PATH+"CICIDS2018_train.csv", index=None)
    pd.DataFrame(data_test).to_csv(BASE_PATH+"CICIDS2018_test.csv", index=None)

def make_small_dataset():
    data = pd.read_csv(BASE_PATH+"CICIDS2018_small_before.csv")
    data['Label'].replace(['Benign', 'Bot', 'Brute Force -Web', 
                            'Brute Force -XSS', 'DDOS attack-HOIC', 
                            'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP',
                            'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 
                            'DoS attacks-SlowHTTPTest','DoS attacks-Slowloris',
                            'FTP-BruteForce', 'Infilteration', 
                            'SQL Injection', 'SSH-Bruteforce'],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace=True)
    data.to_csv(BASE_PATH+"CICIDS2018_small.csv", index=False)
    # data_train, data_test = train_test_split(data, test_size=0.2)
    # data_train.to_csv(BASE_PATH+"CICIDS2018_small_train.csv", index=False)
    # data_test.to_csv(BASE_PATH+"CICIDS2018_small_test.csv", index=False)


if __name__ == "__main__":
    #drop_columns()
    #make_dataset()
    make_small_dataset()