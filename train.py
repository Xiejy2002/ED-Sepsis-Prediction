import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import shap

word_encoder = 'label'  # 'one_hot'

if (word_encoder == 'one_hot'):
    data = pd.read_csv(r'data\data_one_hot.csv')
    for j in range(2,5):
        for i in range(data.shape[0]):
            ele = data.iloc[i,j]
            ele = ele.strip('[].')
            ele = ele.replace('.',',')
            ele = "[" + ele + "]"
            data.iloc[i,j] = ele
        data.iloc[:,j] = data.iloc[:,j].apply(eval)
else:
    data = pd.read_csv(r'data\data_label.csv')

num = data.shape[0]
data = shuffle(data,n_samples=num)
data_trainval = data.iloc[0:int(num*0.9),:]
num_trainval = data_trainval.shape[0]

data_test = data.iloc[int(num*0.9):num,:]
test_fold = random.randint(0,9)

total_TP = 0
total_TN = 0
total_FN = 0
total_FP = 0
sepsis_threshold = 0.5  # pred>threshold -> sepsis

K = 10
for fold in range(K):
    print(f"fold {fold}")
    
    val = data_trainval.iloc[int(fold*num_trainval/K): int((fold+1)*num_trainval/K)]
    train1 = data_trainval.iloc[0: int(fold*num_trainval/K)]
    train2 = data_trainval.iloc[int((fold+1)*num_trainval/K): num_trainval]
    train = pd.concat([train1, train2], axis = 0)

    feature_columns = ['ed_hour','age','gender','race','arrival_transport','temperature','heartrate','resprate','o2sat','sbp','dbp','map','shock index','acuity']
    target_column = 'label'
    xgtrain = xgb.DMatrix(train[feature_columns].values, train[target_column].values)  #TODO: one hot
    xgval = xgb.DMatrix(val[feature_columns].values, val[target_column].values)

    #param = {'max_depth':5, 'eta':0.1, 'silent':1, 'subsample':0.7, 'colsample_bytree':0.7, 'objective':'binary:logistic' }
    param = {'max_depth':5, 'min_child_weight':3, 'eta':0.07, 'gamma':0.3, 'subsample':0.6, 'colsample_bytree':0.7, 'alpha':1, 'lambda':2, 'objective':'binary:logistic' }

    num_round = 150
    bst = xgb.train(param, xgtrain, num_round)
    #watchlist  = [(xgval,'eval'), (xgtrain,'train')]
    #bst = xgb.train(param, xgtrain, num_round, watchlist)

    preds = bst.predict(xgval)
    labels = xgval.get_label()
    TP = sum(1 for i in range(len(preds)) if int(preds[i]>=sepsis_threshold) and labels[i])
    TN = sum(1 for i in range(len(preds)) if int(preds[i]<sepsis_threshold) and not labels[i])
    FP = sum(1 for i in range(len(preds)) if int(preds[i]>=sepsis_threshold) and not labels[i])
    FN = sum(1 for i in range(len(preds)) if int(preds[i]<sepsis_threshold) and labels[i])
    total_TP += TP
    total_FN += FN
    total_TN += TN
    total_FP += FP
    print ('error rate: %f' % (float(FP+FN) / float(TP+TN+FP+FN)))
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    roc_auc = metrics.auc(fpr,tpr)
    confusion_mat=metrics.confusion_matrix(labels, np.rint(preds))

    # interpretation
    plt.figure()
    #xgb.plot_importance(bst, title='feature importance', xlabel='score', ylabel='feature', importance_type='gain')
    plt.bar(range(len(feature_columns)), bst.get_score(importance_type='gain').values())
    #print(bst.get_score(importance_type='gain'))
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=-90, fontsize=10)
    plt.title('Feature importance', fontsize=14)
    plt.savefig('figure/importance_fold{}.pdf'.format(fold), bbox_inches='tight')
    plt.close()

    """ if (fold==test_fold):
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(data[feature_columns])
        #shap_values2 = explainer(data[feature_columns]) 
        #y_base = explainer.expected_value  # logit of sepsis?
        shap.summary_plot(shap_values, data[feature_columns])
        #shap.summary_plot(shap_values, data[feature_columns], plot_type="bar")
        #shap.plots.bar(shap_values2[100], show_data=True)
        #shap.dependence_plot('shock index', shap_values, data[feature_columns], interaction_index=None, show=True) """

    # plotting
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
    plt.savefig('figure/auc_roc_fold{}.pdf'.format(fold))
    plt.close()

    plt.figure()
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(1)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('figure/confusion_matrix_fold{}.pdf'.format(fold))
    plt.close()

    bst.save_model('model/fold{}.model'.format(fold))

    if (fold == test_fold):
        xgtest = xgb.DMatrix(data_test[feature_columns].values, data_test[target_column].values)
        test_preds = bst.predict(xgtest)
        test_labels = xgtest.get_label()
        test_TP = sum(1 for i in range(len(test_preds)) if int(test_preds[i]>=sepsis_threshold) and test_labels[i])
        test_TN = sum(1 for i in range(len(test_preds)) if int(test_preds[i]<sepsis_threshold) and not test_labels[i])
        test_FP = sum(1 for i in range(len(test_preds)) if int(test_preds[i]>=sepsis_threshold) and not test_labels[i])
        test_FN = sum(1 for i in range(len(test_preds)) if int(test_preds[i]<sepsis_threshold) and test_labels[i])

        test_fpr, test_tpr, test_thresholds = metrics.roc_curve(test_labels, test_preds, pos_label=1)
        test_roc_auc = metrics.auc(test_fpr,test_tpr)
        plt.figure()
        lw = 2
        plt.plot(test_fpr,test_tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % test_roc_auc,)
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.savefig('figure/test_auc_roc_fold{}.pdf'.format(fold))
        plt.close()

print ('total TP: %d' % (total_TP))
print ('total FP: %d' % (total_FP))
print ('total TN: %d' % (total_TN))
print ('total FN: %d' % (total_FN))
print ('total error rate: %f' % (float(total_FP+total_FN) / float(total_TP+total_TN+total_FP+total_FN)))
print ('total sepsis error rate: %f' % (float(total_FN) / float(total_TP+total_FN)))
print ('total negative error rate: %f' % (float(total_FP) / float(total_TN+total_FP)))

print ('test TP: %d' % (test_TP))
print ('test FP: %d' % (test_FP))
print ('test TN: %d' % (test_TN))
print ('test FN: %d' % (test_FN))
print ('test error rate: %f' % (float(test_FP+test_FN) / float(test_TP+test_TN+test_FP+test_FN)))
print ('test sepsis error rate: %f' % (float(test_FN) / float(test_TP+test_FN)))
print ('test negative error rate: %f' % (float(test_FP) / float(test_TN+test_FP)))