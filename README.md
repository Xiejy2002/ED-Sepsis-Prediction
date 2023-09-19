# ED-Sepsis-Prediction
Predict whether the patient will have a sepsis onset within 24 hours after ED entrance, based on ED data obtained from mimic iv

数据处理：
由于ICU与ED中的stay_id含义不同，需要进行对齐：在sepsis表格中，对每个(icu)stay_id，找到对应的subject_id，到icustays表格中找到相应的hadm_id，从而在edstays中找到(ed)stay_id，进一步从ED相关表格中得到对应信息

align.py：将ED与ICU数据对齐，生成入ED后24小时内sepsis发病的患者及相关特征的表格：subject_id, hadm_id, in_time, sofa_time, sofa, ed_hour, age, gender, race, arrival_transport, temperature, heartrate, resprate, o2sat, sbp, dbp, map, shock index, acuity, complaint
cleaning.py：对生成表格数据进行清理，去除temperature, heartrate, resprate, o2sat, sbp, dbp中有≥4个数据缺失的行，将race统一为white, black, asain, latin, other，arrival_transport中的unknown改为other
negative.py：从ED数据中，随机挑选出基本等量的未得过sepsis的患者作为负类，且保证其acuity分布基本相同，并提前去除了数据缺失过多的行
combinging.py：将正负类合并为最终使用的数据，并对字符串形式的数据进行encoding
train.py：训练XGBoost模型，并输出相关的分析图像
tune.py：网格调参，对模型性能影响不大

注意triage数据直接从数据库导出后存在问题：有部分乱码，且一些pain和main complaint存在引号问题，会影响后续读取，目前已经调整（在excel中查看错误出现的“第12行”，即可找到并解决相应问题）
