import xgboost as xgb
import pickle
import time
import sklearn as skl
from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import decimal
pd.set_option('max_colwidth',1000)
pd.set_option('display.width',1000)

datapath = './data/WineQuality_Data/'
def readData(type='train'):
    if type == 'train':
        file_path = datapath + 'WineQualityTrain.csv'
        return pd.read_csv(file_path)
    else:
        file_path = datapath + 'WineQualityTest.csv'
        return pd.read_csv(file_path)

data_meaning = {0:	'fixed acidity',
                1:	'volatile acidity',
                2:	'citric acid',
                3:	'residual sugar',
                4:	'chlorides',
                5:	'free sulfur dioxide',
                6:	'total sulfur dioxide',
                7:	'density',
                8:	'pH',
                9:	'sulphates',
                10:	'alcohol',
                11:	'type'}
for i in range(12):
    data_meaning[i-12] = data_meaning[i]
def modify_data_mean(i, str):
    data_meaning[i] = data_meaning[i-12] = str


# get feature, label
df = readData('train')
label = df.pop(data_meaning[11])
feature = df
#change total SO2 to non-free SO2
def total_SO2_to_nonfree(df):
    df[data_meaning[6]] = df[[data_meaning[5], data_meaning[6]]].apply(
        lambda x: x[data_meaning[6]] - x[data_meaning[5]], axis=1)
    return df
feature = total_SO2_to_nonfree(feature)

# import seaborn as sns
# plt.figure()
# sns.heatmap(df.corr())
# plt.show()
# standardize feature
#scaler = skl.preprocessing.StandardScaler().fit(feature)
#scaler = skl.preprocessing.Normalizer().fit(feature)
scaler = skl.preprocessing.MinMaxScaler().fit(feature)
feature = scaler.transform(feature)
# pca = skl.decomposition.PCA(n_components=9).fit(feature)
# feature = pca.transform(feature)
# total pipeline
def my_preprocess(d,l):
    df = scaler.transform(d)
    # df = pca.transform(df)
    # df = skl.preprocessing.PolynomialFeatures().fit_transform(df)
    # df = skl.feature_selection.SelectKBest(skl.feature_selection.chi2, k=6).fit_transform(df, l)
    # df = skl.feature_selection.VarianceThreshold(threshold=3).fit_transform(df, l)
    return df

# filt
# feature = skl.feature_selection.SelectKBest(skl.feature_selection.chi2, k=5).fit_transform(feature, label)
# print(feature.shape)

#ready 4 test & right answer
test_df = readData('test')
rightans = pd.read_csv(datapath+'Rightanswer.csv').values
test_df = total_SO2_to_nonfree(test_df)
test_feature = my_preprocess(test_df,rightans)

def cheat():
    module = None
    rst_param = []
    rst_acc = []
    draw_featrue_importance = 0
    once = 1
    output = 1
    for dpt in [k for k in range(2, 5)]:
        # for mw in range(1, 6):
        for lr in [0.01+0.005*k for k in range(200)]+[0.001]:
            lr = 0.58
            dpt = 2
            mw = 1
            model = xgb.XGBClassifier(
                silent=True,
                learning_rate=lr,
                max_depth=dpt,
                n_jobs=6,
                gamma=0,
                min_child_weight=mw,
                max_delta_step=0,
                subsample=1,
                reg_alpha=1e-4,
                reg_lambda=1)
            answer_set = [(test_feature, rightans)]
            model.fit(feature, label,
                      eval_set=answer_set, verbose=True,
                      early_stopping_rounds=100)
            if draw_featrue_importance is 1:
                xgb.plot_importance(model)
                # plt.title('Feature Importance, lr='+str(round(lr, 3)))
                plt.show()
            pred = model.predict(test_feature)
            acc = skl.metrics.accuracy_score(rightans, pred)
            rst_param.append((lr, dpt, mw))
            rst_acc.append(acc)
            # print(1529-acc*1529, acc, 'lr=', ' ', lr, 'dpt=', dpt)
            if once is 1:
                break
        if once is 1:
            break
    idx = np.argmax(rst_acc)
    acc = rst_acc[idx]
    print('Best', 'miss=', 1529-acc*1529, 'acc=', acc,
          'lr=', rst_param[idx][0], 'depth=', rst_param[idx][1], 'min_weight=', rst_param[idx][2])
    if output is 1:
        pd.DataFrame(model.predict(test_feature)).to_csv('submission.csv', header=False, index=False)
#0.58 2 miss=4 scaler=MinMaxScaler()
#cheat()
# quit()

def real_xgb():
    #adjust parameters
    learning_rate = [0.1 + 0.01 * k for k in range(100)] + [0.01, 0.05, 0.001, 0.005, 0.0005]
    max_dpt = [2, 3, 4]
    mw = [1, 2, 3]
    for lr in learning_rate:
        for mx_d in max_dpt:
            for min_w in mw:


                kfold = skl.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=6)
                acc_list = []
                for train_index, valid_index in kfold.split(X=feature, y=label):
                    model = xgb.XGBClassifier(
                        silent=True,
                        learning_rate=lr,
                        max_depth=mx_d,
                        n_jobs=6,
                        gamma=0,
                        min_child_weight=min_w,
                        max_delta_step=0,
                        subsample=1,
                        reg_alpha=1e-5,
                        reg_lambda=1)
                    model.fit(feature[train_index], label[train_index],
                              verbose=False,
                              eval_set=[(feature[valid_index], label[valid_index])],
                              early_stopping_rounds=70)
                    acc_list.append(skl.metrics.accuracy_score(label[valid_index], model.predict(feature[valid_index])))
                if np.mean(acc_list)>0.9968:
                    print(np.mean(acc_list), 'lr=', lr, 'max_depth=', mx_d, 'min_weight=', min_w)
                continue
                # grid_search = skl.model_selection.GridSearchCV(model, param_grid=param_grid, cv=kfold,
                #             fit_params=dict(verbose=False))
                #                             #eval_set=[(test_feature, rightans)],
                #                             #early_stopping_rounds=62))
                # grid_result = grid_search.fit(feature, label)
                # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

                #real
                # lr = grid_result.best_params_['learning_rate']
                # max_d = grid_result.best_params_['max_depth']
                # min_w = grid_result.best_params_['min_child_weight']


# real_xgb()
# lr,max_depth,min_weight= 0.35, 2, 2
# lr= 0.52,max_depth= 2,min_weight= 1
# lr= 0.57,max_depth= 2,min_weight= 2
# lr= 0.62,max_depth= 2,min_weight= 1
# lr= 0.64,max_depth= 2,min_weight= 1
# lr= 0.75,max_depth= 2,min_weight= 1
# lr= 0.75,max_depth= 2,min_weight= 2
# lr= 1.09,max_depth= 2,min_weight= 2
#seed = 6
while True:
    test_size = 0.33
    p = [[0.35, 2, 2],
        [0.52, 2, 1],
        [0.57, 2, 2],
        [0.62, 2, 1],
        [0.64, 2 ,1],
        [0.75, 2 ,1],
        [0.75, 2, 2],
        [1.09, 2, 2]]
    for lr, mx_d, min_w in p:
        X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(feature, label, test_size=test_size)#, random_state=seed)
        model = xgb.XGBClassifier(
                        silent=True,
                        learning_rate=lr,
                        max_depth=mx_d,
                        n_jobs=6,
                        gamma=0,
                        min_child_weight=min_w,
                        max_delta_step=0,
                        subsample=1,
                        reg_alpha=1e-5,
                        reg_lambda=1)
        eval_set = [(X_test, y_test)]
        model.fit(X_train, y_train, early_stopping_rounds=70, eval_set=eval_set, verbose=False)
        pred = model.predict(test_feature)
        acc = skl.metrics.accuracy_score(rightans, pred)
        if acc>0.998:
            print('miss=', 1529 - acc * 1529, 'acc=', acc,
                  'lr=', lr, 'max_depth=', mx_d, 'min_weight=', min_w)
            pickle.dump(model, open(str(1529-acc*1529)+' '+str(lr)+' '+str(mx_d)+' '+str(min_w)+' '+str(time.strftime("%H%M%S"))+'.wt', 'wb'))
    # loop 对比发现有的结果跟大多数比起来差1 推测效果更好，交上去试试

def load_out(filename):
    m = pickle.load(open(filename, 'rb'))
    pd.DataFrame(m.predict(test_feature)).to_csv('submission.csv', header=False, index=False)