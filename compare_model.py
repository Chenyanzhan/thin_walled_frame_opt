import xgboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


data = pd.read_csv('./引擎盖数据集.csv')
feature_name = data.columns.tolist()
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data=data,columns=feature_name)
X = data.iloc[:,0:11].values
# def feature_scalling(X):
#     mmin = X.min()
#     mmax = X.max()
#     return (X - mmin) / (mmax - mmin), mmin, mmax
#
# X,xmin,xmax = feature_scalling(X)





class othermodel():
    def __init__(self):
        self.path = './'

    def LR_model(self):
        for i in range(7):
            Y = data.iloc[:, 11 + i].values
            Y, ymin, ymax = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))

    def KNN_model(self):
        for i in range(7):
            Y = data.iloc[:, 11 + i].values
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = KNeighborsRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))

    def SVR_model(self):
        for i in range(7):
            Y = data.iloc[:, 11 + i].values
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = SVR()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))

    def RF_model(self):
        for i in range(7):
            Y = data.iloc[:, 11 + i].values
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))

    def ourmodel(self):
        target_list = ['none','stiffness','first mode','weight']
        for i in range(4):
            Y = data.iloc[:, 11 + i].values
            print(Y[0:5])
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = xgboost.XGBRegressor()
            model.fit(X_train, y_train)

            y_true = y_test[0:990]
            y_pred = model.predict(X_test[0:990,:])
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
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.tick_params(top='on',right='on')
            plt.rcParams['font.size'] = 18
            cm = plt.cm.get_cmap('CMRmap_r')
            sc = plt.scatter(y_true, y_pred, s=5, label='Data Points', c=T, alpha=1, cmap=cm)
            plt.plot(x, y, c='#000000ff', linewidth=2, label='Predicted=Target')
            # plt.legend()
            # plt.xlabel('Experimental mode(Hz)', size=15)
            # plt.ylabel('Predicted mode(Hz)', size=15)
            # plt.title('Hybrid model for predicting mode')
            x_name = 'Experimental ' + target_list[i]
            y_name = 'Predicted ' + target_list[i]
            plt.xlabel(x_name, size=18)
            plt.ylabel(y_name, size=18)
            # plt.title('GBDBN for predicting '+ target_list[i])
            # plt.xlim(min_flage, max_flage)
            # plt.ylim(min_flage, max_flage)
            plt.grid(ls='--')
            plt.colorbar(sc)
            plt.show()
            break




other = othermodel()
other.ourmodel()
# other.KNN_model()
# other.SVR_model()
# other.RF_model()
