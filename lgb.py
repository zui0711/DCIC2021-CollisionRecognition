import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time
from tqdm import tqdm

label = pd.read_csv('../raw_data/train_labels.csv')
label['CollectTime'] = pd.to_datetime(label['CollectTime'])

ori_feats = ['加速踏板位置', '车速', '整车当前总电流']


# 预处理
def preprocess(df):
    df = df.sort_values(by=['采集时间']).reset_index(drop=True)
    df.loc[df['车辆行驶里程'] < 10, '车辆行驶里程'] = np.nan
    df.loc[df['车辆行驶里程'] > 1e6, '车辆行驶里程'] = np.nan
    df['车辆行驶里程'] = df['车辆行驶里程'].fillna(method='ffill')
    df['采集时间'] = pd.to_datetime(df['采集时间'])
    return df


# 提取特征
def get_feats(df):
    df = df.reset_index(drop=True)
    df['功率'] = df['整车当前总电流'] * df['整车当前总电压']
    df['车速差'] = df['车速'].diff()
    df['整车当前总电流差'] = df['整车当前总电流'].diff()
    df['方向盘转角差'] = df['方向盘转角'].diff()
    feats = []
    for f in ori_feats+['功率', '车速差', '整车当前总电流差', '方向盘转角差']:
        feats.extend([df[f].mean(), df[f].median(), df[f].std(), df[f].min(), df[f].max()])
    feats.append(df['车辆行驶里程'].mean())

    count = np.array([df.loc[df['电池包主负继电器状态']==name, '车号'].count() for name in ['断开', '连接']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['电池包主正继电器状态']==name, '车号'].count() for name in ['断开', '连接', '粘连']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['制动踏板状态']==name, '车号'].count() for name in ['未踩', '踩下']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['驾驶员离开提示']==name, '车号'].count() for name in ['No Warning', 'Warning(下车时请拔掉钥匙)']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['驾驶员离开提示'] == name, '车号'].count() for name in ['No Warning', 'Warning(下车时请拔掉钥匙)']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['主驾驶座占用状态']==name, '车号'].count() for name in ['空置', '有人', '传感器故障']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['驾驶员安全带状态']==name, '车号'].count() for name in ['未系', '已系']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['手刹状态']==name, '车号'].count() for name in ['手刹放下', '手刹拉起']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['整车钥匙状态']==name, '车号'].count() for name in ['OFF', 'ON', 'ACC']])
    feats.extend(count / np.sum(count))
    count = np.array([df.loc[df['整车当前档位状态']==name, '车号'].count() for name in ['空档', '前进', '后退', '驻车']])
    feats.extend(count / np.sum(count))

    return feats


win = 8
train = []
test = []
new_label = []
test_list = []
test_time = []
print('\nTrain data...\n')
for i in tqdm(range(1, 121)):
    if i == 19 or i == 94:  # 异常值
        continue
    df = pd.read_csv('../raw_data/train/%d.csv' % i)
    df = preprocess(df)
    f_all = get_feats(df)
    if label.loc[i - 1, 'Label'] == 1:
        idx = np.abs(df['采集时间']-label.loc[i - 1, 'CollectTime']).argmin()
        # if idx > 0:
        new_label.append(1)
        feats = get_feats(df.loc[idx-win:idx]) + get_feats(df.loc[idx:idx+win]) + get_feats(df.loc[idx-2:idx+2]) + f_all
        train.append(feats)
    else:
        # 根据规则筛选可能的碰撞时刻
        df['电池包主负继电器状态1'] = df['电池包主负继电器状态'].shift(1)
        df['电池包主负继电器状态2'] = df['电池包主负继电器状态'].shift(2)
        df['整车当前总电流1'] = df['整车当前总电流'].shift(1)
        idx = df[((df['电池包主负继电器状态1']=='连接') & (df['电池包主负继电器状态']=='断开')) |
                 ((df['电池包主负继电器状态2']=='连接') & (df['电池包主负继电器状态1']=='断开'))|
                 ((df['整车当前总电流1'] != 0) & (df['整车当前总电流'] == 0)) ].index
        for iidx in idx:
            feats = get_feats(df.loc[iidx-win:iidx]) + get_feats(df.loc[iidx:iidx+win]) + get_feats(df.loc[iidx-2:iidx+2]) + f_all
            train.append(feats)
            new_label.append(0)

print('\nTest data...\n')
for i in tqdm(range(121, 261)):
    df = pd.read_csv('../raw_data/test_allv2/%d.csv' % i)
    df = preprocess(df)
    df['电池包主负继电器状态1'] = df['电池包主负继电器状态'].shift(1)
    df['电池包主负继电器状态2'] = df['电池包主负继电器状态'].shift(2)
    df['整车当前总电流1'] = df['整车当前总电流'].shift(1)
    f_all = get_feats(df)
    df['电池包主负继电器状态3'] = df['电池包主负继电器状态'].shift(2)
    idx = df[((df['电池包主负继电器状态1'] == '连接') & (df['电池包主负继电器状态'] == '断开')) |
             ((df['电池包主负继电器状态2'] == '连接') & (df['电池包主负继电器状态1'] == '断开')) |
             ((df['整车当前总电流1'] != 0) & (df['整车当前总电流'] == 0)) ].index

    for iidx in idx:
        feats = get_feats(df.loc[iidx-win:iidx]) + get_feats(df.loc[iidx:iidx+win]) + get_feats(df.loc[iidx-2:iidx+2]) + f_all
        test.append(feats)
        test_list.append(i)
        test_time.append(df.loc[iidx, '采集时间'])


train = np.array(train)
test = np.array(test)
new_label = np.array(new_label)
print(train.shape, test.shape)
preds = 0
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.05,
    'metric': 'auc',
    'seed': 2,
    'verbose': -1,
    'nthread': -1,
    'is_unbalance': True,
    }

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=22222)
oof = np.zeros(len(train))
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train, new_label)):
    train_x = train[trn_idx]
    train_y = new_label[trn_idx]
    val_x = train[val_idx]
    val_y = new_label[val_idx]
    print(train_x.shape, val_x.shape)
    train_matrix = lgb.Dataset(train_x, label=train_y)
    val_matrix = lgb.Dataset(val_x, label=val_y)

    model = lgb.train(params, train_matrix, num_boost_round=10000, valid_sets=[train_matrix, val_matrix],
                      verbose_eval=0, early_stopping_rounds=50)
    pred_val_y = model.predict(val_x)
    print(roc_auc_score(val_y, pred_val_y))
    oof[val_idx] = model.predict(train[val_idx])
    preds += model.predict(test) / 5

print('OOF', roc_auc_score(new_label, oof))
sub = pd.DataFrame()
sub['车号'] = np.array(range(121, 261))
sub['Label'] = 0
sub['CollectTime'] = np.nan
dic = {}
dic_time = {}
for (x, y, z) in zip(test_list, preds, test_time):
    if x in dic:
        dic[x].append(y)
        dic_time[x].append(z)
    else:
        dic[x] = [y]
        dic_time[x] = [z]

for i in range(121, 261):
    idx = np.argmax(dic[i])
    sub.loc[i-121, 'Label'] = np.max(dic[i])
    if np.max(dic[i]) >= 0.5:
        sub.loc[i - 121, 'CollectTime'] = dic_time[i][idx]  # t
sub['Label'] = sub['Label'].map(lambda x: 1 if x >= 0.5 else 0)
print(len(sub['Label']), np.sum(sub['Label']))

sub.to_csv(time.strftime('../prediction_result/result.csv'), index=False)


