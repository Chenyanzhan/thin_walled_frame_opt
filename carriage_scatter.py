import xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
# f = open("./数据集.csv")
# print(f.encoding)
data = pd.read_excel('./数据集2.xlsx')
print(data.shape)
# feature_name = data.columns.tolist()
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data)
# data = pd.DataFrame(data=data,columns=feature_name)
print(data.shape)

X = data.iloc[:,0:77].values
# X_1 = data.iloc[:,26:77].values
# Y = data.iloc[:,15].values
def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax
# X,xmin,xmax = feature_scalling(X)

name_index = 0
for i in [5,0,2,4]:
    Y = data.iloc[:, 82 + i].values
    Y, ymin, ymax = feature_scalling(Y)
    seed = 7
    test_size = 0.3
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = test_size,random_state = seed)
    # model = xgboost.XGBRegressor()
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
    # print('指标' + str(i + 1))
    # print('rmse: %.6f' % rmse)
    # print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
    # print('r_square: %.6f' % r2_score(y_test, y_pred))

   #保存文件
    target_list = ['none', 'stiffness', 'first mode', 'weight']
    # save_name = './model/'+ 'carriage' + target_list[name_index] +'.pkl'
    # joblib.dump(model,save_name)

    ##绘图
    y_true = Y
    y_pred =model.predict(X)
    print(r2_score(y_true, y_pred))
    print(mean_absolute_error(y_true, y_pred))
    max_flage = max(y_true.max(), y_pred.max())
    min_flage = min(y_true.min(), y_pred.min())
    step = (max_flage - min_flage) / 100
    # x = range(int(max_flage))
    x = np.arange(min_flage, max_flage, step)
    y = x
    # plt.scatter(Y_true, y_result, s=45, label='predict')
    # 定义偏离程度大小
    # T = abs(y_true - y_pred)/y_true
    T = abs(y_true - y_pred)
    plt.axes().set_facecolor('whitesmoke')
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.tick_params(top='on', right='on')
    plt.rcParams['font.size'] = 18
    plt.scatter(y_true, y_pred, s=7, label='Data Points', c='#00aa00ff', alpha=1)
    plt.plot(x, y, c='#ff007fff', linewidth=3, label='Predicted=Target')
    x_name = 'Experimental ' + target_list[name_index]
    y_name = 'Predicted ' + target_list[name_index]
    plt.xlabel(x_name, size=18)
    plt.ylabel(y_name, size=18)
    # plt.xlim(min_flage, max_flage)
    # plt.ylim(min_flage, max_flage)
    plt.grid(ls='--')
    plt.show()

    name_index += 1

