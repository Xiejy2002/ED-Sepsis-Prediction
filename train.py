import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv(r'data\data.csv')
num = data.shape[0]
data = shuffle(data,n_samples=num)

total_error = 0
total_sepsis_error = 0
total_negative_error = 0

K = 10
for fold in range(K):
    print(f"fold {fold}")
    
    test = data.iloc[int(fold*num/K):int((fold+1)*num/K)]
    train1 = data.iloc[0:int(fold*num/K)]
    train2 = data.iloc[int((fold+1)*num/K):num]
    train = pd.concat([train1, train2], axis = 0)

    feature_columns = ['ed_hour','age','gender','race','arrival_transport','temperature','heartrate','resprate','o2sat','sbp','dbp','map','shock index','acuity']
    target_column = 'label'
    xgtrain = xgb.DMatrix(train[feature_columns].values, train[target_column].values)
    xgtest = xgb.DMatrix(test[feature_columns].values, test[target_column].values)

    #参数设定
    #param = {'max_depth':5, 'eta':0.1, 'silent':1, 'subsample':0.7, 'colsample_bytree':0.7, 'objective':'binary:logistic' }
    param = {'max_depth':5, 'min_child_weight':3, 'eta':0.07, 'gamma':0.3, 'subsample':0.6, 'colsample_bytree':0.7, 'alpha':1, 'lambda':2, 'objective':'binary:logistic' }

    num_round = 150
    bst = xgb.train(param, xgtrain, num_round)
    #watchlist  = [(xgtest,'eval'), (xgtrain,'train')]
    #bst = xgb.train(param, xgtrain, num_round, watchlist)

    # 使用模型预测
    preds = bst.predict(xgtest)

    # 判断准确率
    labels = xgtest.get_label()
    total_error += (sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds)))
    #print ('error rate: %f' % (sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
    total_sepsis_error += (sum(1 for i in range(len(preds)) if (preds[i]<0.5 and labels[i])) /822.0)
    #print ('sepsis error rate: %f' % (sum(1 for i in range(len(preds)) if (preds[i]<0.5 and labels[i])) /822.0))
    total_negative_error += (sum(1 for i in range(len(preds)) if (preds[i]>0.5 and not labels[i])) /840.0)
    #print ('negative error rate: %f' % (sum(1 for i in range(len(preds)) if (preds[i]>0.5 and not labels[i])) /840.0))
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    roc_auc = metrics.auc(fpr,tpr)
    confusion_mat=metrics.confusion_matrix(labels, np.rint(preds))

    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % roc_auc,)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig('figure/auc_roc_fold{}.pdf.pdf'.format(fold))

    plt.figure()
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(1)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('figure/confusion_matrix_fold{}.pdf.pdf'.format(fold))

    bst.save_model('model/fold{}.model'.format(fold))

print ('total error rate: %f' % (total_error/10.0))
print ('total sepsis error rate: %f' % (total_sepsis_error/10.0))
print ('total negative error rate: %f' % (total_negative_error/10.0))