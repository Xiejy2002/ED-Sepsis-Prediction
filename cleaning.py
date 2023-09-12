import pandas as pd
import numpy as np
import csv

sepsis = pd.read_csv(r'data\sepsis_24h.csv',sep=',',header='infer').values[0::,0::]
num_sepsis = sepsis.shape[0]
headers = ['subject_id','hadm_id','in_time','sofa_time','sofa','ed_hour','age','gender','race','arrival_transport',
           'temperature','heartrate','resprate','o2sat','sbp','dbp','map','shock index','acuity','complaint']
rows = []

for i in range(num_sepsis):
    row = sepsis[i,:]

    race = row[8]
    if "BLACK" in race: row[8] = "BLACK"
    elif "WHITE" in race: row[8] = "WHITE"
    elif "ASIAN" in race: row[8] = "ASIAN"
    elif "LATINO" in race: row[8] = "LATINO"
    else: row[8] = "OTHER"

    transport = row[9]
    if "UNKNOWN" in transport: row[9] = "OTHER"

    nan_count = 0
    for j in range(6):
        if np.isnan(row[10+j]): nan_count += 1
    if nan_count<4:
        rows.append(row)

with open('data\sepsis_24h_clean.csv','w',encoding='utf8',newline='') as f :
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)    