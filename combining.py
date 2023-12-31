import pandas as pd
import numpy as np
import csv
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

word_encoder = 'label'  # 'one_hot'

sepsis = pd.read_csv(r'data\sepsis_24h_clean.csv',sep=',',header='infer').values[0::,0::]
negative = pd.read_csv(r'data\negative_clean.csv',sep=',',header='infer').values[0::,0::]

num_sepsis = sepsis.shape[0]
num_negative = negative.shape[0]
num_total = num_sepsis + num_negative
headers = ['ed_hour','age','gender','race','arrival_transport','temperature','heartrate','resprate','o2sat','sbp','dbp','map','shock index','acuity','label']
rows = []

for i in range(0,num_sepsis):
    rows.append((sepsis[i,5], sepsis[i,6], sepsis[i,7], sepsis[i,8], sepsis[i,9], sepsis[i,10], sepsis[i,11], sepsis[i,12], sepsis[i,13], sepsis[i,14], sepsis[i,15], sepsis[i,16], sepsis[i,17], sepsis[i,18], 1))
for i in range(0,num_negative):
    rows.append((negative[i,5], negative[i,6], negative[i,7], negative[i,8], negative[i,9], negative[i,10], negative[i,11], negative[i,12], negative[i,13], negative[i,14], negative[i,15], negative[i,16], negative[i,17], negative[i,18], 0))

random.shuffle(rows)
with open('data/data_pre_encode.csv','w',encoding='utf8',newline='') as f :
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

data = pd.read_csv(r'data\data_pre_encode.csv',sep=',',header='infer').values[0::,0::]

if (word_encoder == 'one_hot'):
    one_hot_encoder = OneHotEncoder()
    for i in range(2,5):
        one_hot_encoder = one_hot_encoder.fit(data[:,i].reshape(-1, 1))
        data_encoded = one_hot_encoder.transform(data[:,i].reshape(-1, 1)).toarray()
        for j in range(num_total):
            data[j,i] = data_encoded[j,:]
    with open('data/data_one_hot.csv','w',encoding='utf8',newline='') as f :
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

else:
    label_encoder = LabelEncoder()
    for i in range(2,5):
        label_encoder = label_encoder.fit(data[:,i])
        data[:,i] = label_encoder.transform(data[:,i])
    with open('data/data_label.csv','w',encoding='utf8',newline='') as f :
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

