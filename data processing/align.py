import pandas as pd
import numpy as np
import datetime
import csv
onset = pd.read_csv(r'C:\Users\10465\Desktop\dataset\sepsis_onset.csv',sep=',',header='infer',usecols=[0,1,2,3]).values[0::,0::]
icustay = pd.read_csv(r'C:\Users\10465\Desktop\dataset\icustays.csv',sep=',',header='infer',usecols=[0,1,2]).values[0::,0::]
edstay = pd.read_csv(r'C:\Users\10465\Desktop\dataset\edstays.csv',sep=',',header='infer',usecols=[0,1,2,3,4,5,6,7]).values[0::,0::]
triage = pd.read_csv(r'C:\Users\10465\Desktop\dataset\triage.csv',sep=',',header='infer',usecols=[0,1,2,3,4,5,6,7,8,9,10]).values[0::,0::]
patient = pd.read_csv(r'C:\Users\10465\Desktop\dataset\patient.csv',sep=',',header='infer',usecols=[0,1,2]).values[0::,0::]

num_sepsis = onset.shape[0]
sepsis_24h = 0
headers = ['subject_id','hadm_id','in_time','sofa_time','sofa','ed_hour','age','gender','race','arrival_transport',
           'temperature','heartrate','resprate','o2sat','sbp','dbp','map','shock index','acuity','complaint']
rows = []

for i in range(num_sepsis):
    subject_id = onset[i,0]
    icu_stay_id = onset[i,1]
    onset_time = datetime.datetime.strptime(onset[i,2],'%Y-%m-%d %H:%M:%S')
    sofa_score = onset[i,3]
    icu_row = np.where(icustay == icu_stay_id)[0]
    hadm_id = icustay[icu_row[0],1]
    ed_row = np.where(edstay == hadm_id)[0]

    if (ed_row.size>0):
        ed_stay_id = edstay[ed_row[0],2]
        ed_in_time = datetime.datetime.strptime(edstay[ed_row[0],3],'%Y-%m-%d %H:%M:%S')
        ed_hour = ed_in_time.hour
        patient_row = np.where(patient == subject_id)[0]
        triage_row = np.where(triage == ed_stay_id)[0]

        if ((onset_time-ed_in_time).days<1):
            heartrate = triage[triage_row[0], 3]
            sbp = triage[triage_row[0], 6]
            dbp = triage[triage_row[0], 7]

            if (not np.isnan(heartrate) and not np.isnan(sbp)): shock_index = heartrate/sbp
            else: shock_index = float('nan')
            if (not np.isnan(sbp) and not np.isnan(dbp)): map = (sbp+2*dbp)/3
            else: map = float('nan')

            rows.append((subject_id, hadm_id, ed_in_time.strftime('%Y-%m-%d %H:%M:%S'), onset_time.strftime('%Y-%m-%d %H:%M:%S'), sofa_score, ed_hour, patient[patient_row[0],2], edstay[ed_row[0],5], edstay[ed_row[0],6], edstay[ed_row[0],7], 
                         triage[triage_row[0], 2], heartrate, triage[triage_row[0], 4], triage[triage_row[0], 5], sbp, dbp, map, shock_index, triage[triage_row[0], 9], triage[triage_row[0], 10]))
            sepsis_24h += 1
    print(i)

with open('sepsis_24h.csv','w',encoding='utf8',newline='') as f :
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)