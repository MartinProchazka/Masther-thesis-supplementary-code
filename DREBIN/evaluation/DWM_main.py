
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score,recall_score, precision_score
import random

import classes as cl
import postprocessing_functions as pf

def get_balance_dataset(data,label,downsample=[0,0]):
    out_data=[]
    out_label=[]
    for i in range(len(data)):
        if downsample[0]==0: #downsample benign samples
            if label[i]==1:
                out_data.append(data[i])
                out_label.append(label[i])
            if label[i] == 0 and random.randint(0,downsample[1])==downsample[1]:
                out_data.append(data[i])
                out_label.append(label[i])
        else:
            if random.randint(0,downsample[0])==downsample[0]:
                if label[i]==1:
                    out_data.append(data[i])
                    out_label.append(label[i])
                if label[i] == 0:
                    out_data.append(data[i])
                    out_label.append(label[i])
    return np.array(out_data),np.array(out_label)

def main(args=[42,0,0]):
    seed = args[0]
    downsample = args[1:]
    random.seed(seed)
    np.random.seed(seed)
    CSV_FILE = "drebin_drift.parquet.zip"

    # read data and remove unused collumns, 0 = benign, 1 = malware
    data = pd.read_parquet(CSV_FILE)
    labels = data['label']
    UNUSED_COLUMNS = ["label", "sha256", "submission_date"]
    for c in UNUSED_COLUMNS:
        del data[c]

    #downsample and extract features
    from sklearn.feature_extraction.text import TfidfVectorizer

    data_1,labels_1=get_balance_dataset(np.asarray(data),np.asarray(labels),downsample=downsample)
    X_data = []
    for i in range(10):
        train_data = data_1[:,i]
        tfidf = TfidfVectorizer(max_features=10)
        tfidf.fit(train_data,labels_1) 
        train_tfidf = tfidf.transform(train_data).todense()
        if len(X_data) == 0:
            X_data = train_tfidf
        # concatenate existing features
        else:
            X_data = np.concatenate((X_data, train_tfidf), axis=1)

    #normalize features
    from sklearn.preprocessing import StandardScaler
    X_data = np.array(X_data)
    train_target = np.array(labels_1)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(X_data)
    

    #process dataset
    delay = 1000
    process_usual = cl.DWM(train_data[:1],train_target[:1],50, 0.5, 2, 0.01, delay, 2, smooth=True)
    predictions_usual = []
    for data_index  in range(len(train_data)):
        if data_index < delay:  #we do not have labels for process yet
            predictions_usual.append(process_usual.process_sample(train_data[data_index : data_index + 1],[-1]))
        else:
            predictions_usual.append(process_usual.process_sample(train_data[data_index : data_index + 1],train_target[data_index - delay: data_index - delay + 1]))
    
    #get output statistics
    ret_1 = f1_score(train_target,predictions_usual)
    overall_acc_usuala = accuracy_score(train_target,predictions_usual)
    recol = recall_score(train_target, predictions_usual)
    precision = precision_score(train_target, predictions_usual)

    return (seed,ret_1, overall_acc_usuala,recol,precision)




if __name__ == "__main__":
    main()
