import streamlit as st
from streamlit_shap import st_shap
from sklearn.ensemble import RandomForestClassifier
import shap
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from statistics import mean, stdev
from sklearn.model_selection import KFold
import torch
from PIL import Image

localpath = 'role_detecting/data/'
gitpath = 'data/'

# path = localpath
path = gitpath

df = pd.read_csv(path + 'social_beh_cog_emo_indeces.csv', encoding='utf_8_sig')

rforest = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=0)

# X_exp = df.drop(['group_id', 'group_type', 'name', 'role', 'role_label', '3', '12', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', 'exp_none', 'int_none'], 1)
X_exp = df.drop(['group_id', 'group_type', 'name', 'role', 'role_label', 'exp_none', 'int_none'], 1)
y_exp = df['role_label']

st.title('角色识别随机森林模型可解释分析')

# st.subheader('当前数据集： ')
# st.dataframe(X_exp)

# st.subheader('可视化随机森林结果： ')
# image = Image.open(path + 'rforest_img.png')
# st.image(image, caption='随机森林结果可视化')


st.subheader('当前数据集的描述性统计： ')
st.dataframe(X_exp.describe().T)

rforest.fit(X_exp, y_exp)
_y_pred = rforest.predict(X_exp)
_y = pd.DataFrame(_y_pred)
_y.columns = ['type_code']
# acc = round(accuracy_score(y_exp, _y_pred), 5)

def _accuracy(f, Y_test, X_test):
    acc = accuracy_score(Y_test, f(X_test))
    f1 = f1_score(Y_test, f(X_test), average='macro')
    precision = precision_score(Y_test, f(X_test), average='macro')
    recall = recall_score(Y_test, f(X_test), average='macro')
    return acc, f1, precision, recall

@st.cache(persist=True)
def myfuc(f, X, y):
    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y))
    acc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X):
        X_train, X_test = X[train_index], X[val_index]
        Y_train, Y_test = y[train_index], y[val_index]
        f.fit(X_train, Y_train)
        acc, f1, precision, recall = _accuracy(f.predict, Y_test, X_test)
        acc_list.append(acc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
    acc_mean = mean(acc_list)
    f1_mean = mean(f1_list)
    precision_mean = mean(precision_list)
    recall_mean = mean(recall_list)
    acc_std = np.std(acc_list)
    f1_std = np.std(f1_list)
    precision_std = np.std(precision_list)
    recall_std = np.std(recall_list)
    return {
        'Accuracy_macro': [f'{acc_mean:.3f} ± {acc_std:.3f}'],
        'F1-Score_macro':[f'{f1_mean:.3f} ± {f1_std:.3f}'],
        'Precision_macro':[f'{precision_mean:.3f} ± {precision_std:.3f}'],
        'Recall_macro':[f'{recall_mean:.3f} ± {recall_std:.3f}'],
    }

# model_val_df = pd.DataFrame(myfuc(rforest, X_exp, y_exp))
# st.dataframe(model_val_df)

name_to_id = {
    'A—Ele—马辰歌—探究类': 0,
    'A—Ele—孟繁睿—整合类': 1,
    'A—Ele—王商齐—辅助类': 2,
    'A—Ele—杨铭宇—协调类': 3,
    'A—Ele—赵梓祺—边缘类': 4,
    'A—Lig—马辰歌—辅助类': 5,
    'A—Lig—孟繁睿—整合类': 6,
    'A—Lig—王商齐—辅助类': 7,
    'A—Lig—杨铭宇—探究类': 8,
    'A—Lig—赵梓祺—辅助类': 9,
    'A—Mag—马辰歌—协调类': 10,
    'A—Mag—孟繁睿—整合类': 11,
    'A—Mag—王商齐—辅助类': 12,
    'A—Mag—杨铭宇—探究类': 13,
    'A—Mag—赵梓祺—辅助类': 14,
    'B—Ele—胡天宇—探究类': 15,
    'B—Ele—黎启轩—协调类': 16,
    'B—Ele—孙濡茗—辅助类': 17,
    'BB—Ele—武言—辅助类':   18,
    'B—Ele—张珂函—协调类': 19,
    'B—Ele—郑凯益—整合类': 20,
    'B—Lig—胡天宇—探究类': 21,
    'B—Lig—黎启轩—探究类': 22,
    'B—Lig—孙濡茗—整合类': 23,
    'BB—Lig—武言—辅助类':   24,
    'B—Lig—张珂函—协调类': 25,
    'B—Lig—郑凯益—探究类': 26,
    'B—Mag—胡天宇—探究类': 27,
    'B—Mag—黎启轩—探究类': 28,
    'B—Mag—孙濡茗—边缘类': 29,
    'BB—Mag—武言—辅助类':   30,
    'B—Mag—张珂函—协调类': 31,
    'B—Mag—郑凯益—协调类': 32,
    'C—Ele—陈静丹—边缘类': 33,
    'C—Ele—高迪恩—协调类': 34,
    'C—Ele—刘欣怡—整合类': 35,
    'C—Ele—汤伟喆—边缘类': 36,
    'C—Ele—姚佳一—探究类': 37,
    'C—Lig—高迪恩—协调类': 38,
    'C—Lig—刘欣怡—整合类': 39,
    'C—Lig—汤伟喆—探究类': 40,
    'C—Lig—姚佳一—探究类': 41,
    'C—Mag—陈静丹—整合类': 42,
    'C—Mag—高迪恩—协调类': 43,
    'C—Mag—刘欣怡—探究类': 44,
    'C—Mag—汤伟喆—辅助类': 45,
    'C—Mag—姚佳一—探究类': 46,
    'D—Ele—胡子晗—探究类': 47,
    'D—Ele—姜智皓—协调类': 48,
    'D—Ele—史瑞恩—辅助类': 49,
    'D—Ele—尹子辰—整合类': 50,
    'D—Ele—张峻源—探究类': 51,
    'D—Lig—胡子晗—探究类': 52,
    'D—Lig—姜智皓—协调类': 53,
    'D—Lig—史瑞恩—辅助类': 54,
    'D—Lig—尹子辰—整合类': 55,
    'D—Lig—张峻源—辅助类': 56,
    'D—Mag—胡子晗—探究类': 57,
    'D—Mag—姜智皓—协调类': 58,
    'D—Mag—史瑞恩—辅助类': 59,
    'D—Mag—尹子辰—整合类': 60,
    'D—Mag—张峻源—探究类': 61,
}
type_code_to_name = ['', '协调类', '整合类', '探究类', '辅助类', '边缘类', ]

explainer = shap.TreeExplainer(rforest)

st.subheader('影响模型的重要特征：   ')
_shap_values = explainer.shap_values(X_exp)
st_shap(shap.summary_plot(_shap_values[1], X_exp, plot_type='bar'), height=600, width=800)

# st.write('模型准确率：   ', acc)

st.subheader('模型解释：   ')
user_name = st.selectbox(
    '请选择要查看的样本：',
    name_to_id.keys())

current_id = name_to_id[user_name]

st.write('你当前选择的样本为：   ', user_name)
if user_name:
    _x = X_exp.iloc[current_id, :].astype(float)
    st.write('你当前选择的样本预测结果为：   ', type_code_to_name[_y.loc[current_id, 'type_code']])
    shap_values = explainer.shap_values(_x)
    shap.initjs()
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], _x), height=200, width=800)
