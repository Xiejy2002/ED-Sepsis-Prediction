import pandas as pd
import numpy as np
import datetime
import csv
import random

acuity_sample = False
acuity_1_num = 2500
acuity_2_num = 4800
acuity_3_num = 1100
nagative_num = 8400
temp_threshold = 0  # only take negative sample with temperature above threshold

onset = pd.read_csv(r'data\sepsis_onset.csv',sep=',',header='infer',usecols=[0,1,2,3]).values[0::,0::]
edstay = pd.read_csv(r'data\edstays.csv',sep=',',header='infer',usecols=[0,1,2,3,4,5,6,7]).values[0::,0::]
triage = pd.read_csv(r'data\triage.csv',sep=',',header='infer',usecols=[0,1,2,3,4,5,6,7,8,9,10]).values[0::,0::]
patient = pd.read_csv(r'data\patient.csv',sep=',',header='infer',usecols=[0,1,2]).values[0::,0::]

headers = ['subject_id','hadm_id','in_time','sofa_time','sofa','ed_hour','age','gender','race','arrival_transport',
           'temperature','heartrate','resprate','o2sat','sbp','dbp','map','shock index','acuity','complaint']
rows = []

if (acuity_sample):
    triage_1_num = 0
    list_triage_1 = np.where(triage[:,9] == 1)[0]
    index_triage_1 = np.random.choice(list_triage_1, len(list_triage_1), replace=False)
    for i in range(len(index_triage_1)):
        index = index_triage_1[i]
        if ((triage[index,0] not in onset[:,0]) and (triage[index,2]>=temp_threshold)):
            subject_id = triage[index,0]
            ed_stay_id = triage[index,1]
            patient_row = np.where(patient == subject_id)[0]
            ed_row = np.where(edstay == ed_stay_id)[0]
            ed_in_time = datetime.datetime.strptime(edstay[ed_row[0],3],'%Y-%m-%d %H:%M:%S')
            ed_hour = ed_in_time.hour

            heartrate = triage[index, 3]
            sbp = triage[index, 6]
            dbp = triage[index, 7]
            if (not np.isnan(heartrate) and not np.isnan(sbp)): shock_index = heartrate/sbp
            else: shock_index = float('nan')
            if (not np.isnan(sbp) and not np.isnan(dbp)): map = (sbp+2*dbp)/3
            else: map = float('nan')

            nan_count = 0
            for j in range(6):
                if np.isnan(triage[index,2+j]): nan_count += 1
            if nan_count<4:
                rows.append((subject_id, edstay[ed_row[0],1], ed_in_time.strftime('%Y-%m-%d %H:%M:%S'), float('nan'), float('nan'), ed_hour, patient[patient_row[0],2], edstay[ed_row[0],5], edstay[ed_row[0],6], edstay[ed_row[0],7], 
                            triage[index, 2], heartrate, triage[index, 4], triage[index, 5], sbp, dbp, map, shock_index, triage[index, 9], triage[index, 10]))
                triage_1_num += 1

        #print(i, triage_1_num)
        if (triage_1_num == acuity_1_num): break

    triage_2_num = 0
    list_triage_2 = np.where(triage[:,9] == 2)[0]
    index_triage_2 = np.random.choice(list_triage_2, len(list_triage_2), replace=False)
    for i in range(len(index_triage_2)):
        index = index_triage_2[i]
        if ((triage[index,0] not in onset[:,0]) and (triage[index,2]>=temp_threshold)):
            subject_id = triage[index,0]
            ed_stay_id = triage[index,1]
            patient_row = np.where(patient == subject_id)[0]
            ed_row = np.where(edstay == ed_stay_id)[0]
            ed_in_time = datetime.datetime.strptime(edstay[ed_row[0],3],'%Y-%m-%d %H:%M:%S')
            ed_hour = ed_in_time.hour

            heartrate = triage[index, 3]
            sbp = triage[index, 6]
            dbp = triage[index, 7]
            if (not np.isnan(heartrate) and not np.isnan(sbp)): shock_index = heartrate/sbp
            else: shock_index = float('nan')
            if (not np.isnan(sbp) and not np.isnan(dbp)): map = (sbp+2*dbp)/3
            else: map = float('nan')

            nan_count = 0
            for j in range(6):
                if np.isnan(triage[index,2+j]): nan_count += 1
            if nan_count<4:
                rows.append((subject_id, edstay[ed_row[0],1], ed_in_time.strftime('%Y-%m-%d %H:%M:%S'), float('nan'), float('nan'), ed_hour, patient[patient_row[0],2], edstay[ed_row[0],5], edstay[ed_row[0],6], edstay[ed_row[0],7], 
                            triage[index, 2], heartrate, triage[index, 4], triage[index, 5], sbp, dbp, map, shock_index, triage[index, 9], triage[index, 10]))
                triage_2_num += 1
            
        #print(i, triage_2_num)
        if (triage_2_num == acuity_2_num): break

    triage_3_num = 0
    list_triage_3 = np.where(triage[:,9] == 3)[0]
    index_triage_3 = np.random.choice(list_triage_3, len(list_triage_3), replace=False)
    for i in range(len(index_triage_3)):
        index = index_triage_3[i]
        if ((triage[index,0] not in onset[:,0]) and (triage[index,2]>=temp_threshold)):
            subject_id = triage[index,0]
            ed_stay_id = triage[index,1]
            patient_row = np.where(patient == subject_id)[0]
            ed_row = np.where(edstay == ed_stay_id)[0]
            ed_in_time = datetime.datetime.strptime(edstay[ed_row[0],3],'%Y-%m-%d %H:%M:%S')
            ed_hour = ed_in_time.hour

            heartrate = triage[index, 3]
            sbp = triage[index, 6]
            dbp = triage[index, 7]
            if (not np.isnan(heartrate) and not np.isnan(sbp)): shock_index = heartrate/sbp
            else: shock_index = float('nan')
            if (not np.isnan(sbp) and not np.isnan(dbp)): map = (sbp+2*dbp)/3
            else: map = float('nan')

            nan_count = 0
            for j in range(6):
                if np.isnan(triage[index,2+j]): nan_count += 1
            if nan_count<4:
                rows.append((subject_id, edstay[ed_row[0],1], ed_in_time.strftime('%Y-%m-%d %H:%M:%S'), float('nan'), float('nan'), ed_hour, patient[patient_row[0],2], edstay[ed_row[0],5], edstay[ed_row[0],6], edstay[ed_row[0],7], 
                            triage[index, 2], heartrate, triage[index, 4], triage[index, 5], sbp, dbp, map, shock_index, triage[index, 9], triage[index, 10]))
                triage_3_num += 1

        #print(i, triage_3_num)
        if (triage_3_num == acuity_3_num): break

else:
    triage_num = 0
    index_triage = np.random.choice(triage.shape[0], triage.shape[0], replace=False)
    for i in range(len(index_triage)):
        index = index_triage[i]
        if ((triage[index,0] not in onset[:,0]) and (triage[index,2]>=temp_threshold)):
            subject_id = triage[index,0]
            ed_stay_id = triage[index,1]
            patient_row = np.where(patient == subject_id)[0]
            ed_row = np.where(edstay == ed_stay_id)[0]
            ed_in_time = datetime.datetime.strptime(edstay[ed_row[0],3],'%Y-%m-%d %H:%M:%S')
            ed_hour = ed_in_time.hour

            heartrate = triage[index, 3]
            sbp = triage[index, 6]
            dbp = triage[index, 7]
            if (not np.isnan(heartrate) and not np.isnan(sbp)): shock_index = heartrate/sbp
            else: shock_index = float('nan')
            if (not np.isnan(sbp) and not np.isnan(dbp)): map = (sbp+2*dbp)/3
            else: map = float('nan')

            nan_count = 0
            for j in range(6):
                if np.isnan(triage[index,2+j]): nan_count += 1
            if nan_count<4:
                rows.append((subject_id, edstay[ed_row[0],1], ed_in_time.strftime('%Y-%m-%d %H:%M:%S'), float('nan'), float('nan'), ed_hour, patient[patient_row[0],2], edstay[ed_row[0],5], edstay[ed_row[0],6], edstay[ed_row[0],7], 
                            triage[index, 2], heartrate, triage[index, 4], triage[index, 5], sbp, dbp, map, shock_index, triage[index, 9], triage[index, 10]))
                triage_num += 1

        #print(i, triage_num)
        if (triage_num == nagative_num): break

random.shuffle(rows)
with open('data/negative.csv','w',encoding='utf8',newline='') as f :
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)