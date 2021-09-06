# -*- enconding: utf-8 -*-
# @ModuleName: deep_forest_prediction
# @Function:
# @Author: Yanzhan Chen
# @Time: 2021/6/18 10:15
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
import pandas as pd
import numpy as np
from deepforest import CascadeForestRegressor
import matplotlib.pyplot as plt


class deepforest_model():
    def __init__(self,path):
        if path == './carriage.csv':
            self.case = 'carriage'
            self.start = 0
            self.end = 77
            self.mass = 82
            self.torsion = 84
            self.mode = 86
        elif path == './chassis.csv':
            self.case = 'chassis'
            self.start = 3
            self.end = 10
            self.mass = 13
            self.torsion = 12 #后扭刚度
            self.mode = 10  #弯曲模态
        elif path == './hood.xls':
            self.case = 'hood'
            self.start = 3
            self.end = 14
            self.mass = 22
            self.torsion = 20 #扭转刚度
            self.mode = 16  #一阶模态

        elif path == './hood_sample.csv':
            self.case = 'hood'
            self.start = 0
            self.end = 11
            self.mass = 18
            self.torsion = 16 #扭转刚度
            self.mode = 12  #一阶模态

        self.path = path
        print('load the data:'+self.path)




    def train(self,target):
        print('the predictive label is:'+target)

        if self.path == './hood.xls':
            data = pd.read_excel(self.path)
        else:
            data = pd.read_csv(self.path)

        X = data.iloc[:, self.start:self.end].values
        if target == 'mass':
            y = data.iloc[:, self.mass].values
        if target == 'torsional stiffness':
            y = data.iloc[:, self.torsion].values
        if target == 'first mode':
            y = data.iloc[:, self.mode].values
        seed = 7
        test_size = 0.10
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state = seed)
        model = CascadeForestRegressor()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
        print('rmse: %.6f'%rmse)
        print('mae: %.6f'%mean_absolute_error(y_pred,y_test))
        print('r_square: %.6f'%r2_score(y_test,y_pred))

        ### Adjusted_R2
        # n = X_test.shape[0]
        # print('The total samples are %d'%n)
        # p = self.end - self.start
        # R2 = r2_score(y_test,y_pred)
        # print('Adjusted_R2: %0.6f' %(1-((1-R2*R2)*(n-1))/(n-p-1)))

        # model.save(self.case+target)
        # print('save the model successfully!')

        ##绘图
        for i in range(2):
            y_true = y
            y_pred = model.predict(X)
            max_flage = max(y_true.max(), y_pred.max())
            min_flage = min(y_true.min(), y_pred.min())
            step = (max_flage - min_flage) / 100
            # x = range(int(max_flage))
            x = np.arange(min_flage, max_flage, step)
            y_hat = x
            # plt.scatter(Y_true, y_result, s=45, label='predict')
            # 定义偏离程度大小
            # T = abs(y_true - y_pred)/y_true
            T = abs(y_true - y_pred)
            plt.rcParams['figure.figsize'] = (3,3)
            plt.axes().set_facecolor('whitesmoke')
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.tick_params(top='on', right='on')
            plt.rcParams['font.size'] = 10
            plt.scatter(y_true, y_pred, s=7, label='Data Points', c='#00aa00ff', alpha=1)
            plt.plot(x, y_hat, c='#ff007fff', linewidth=3, label='Predicted=Target')
            x_name = 'Real ' + target + '(Hz)'
            # x_name = 'Real ' + 'bending stiffness' + '(N·m/deg)'
            y_name = 'Predicted ' + target + '(Hz)'
            # y_name = 'Predicted ' + 'bending stiffness' + '(N·m/deg)'
            plt.xlabel(x_name, size=10)
            plt.ylabel(y_name, size=10)
            plt.legend()
            plt.xlim(min_flage, max_flage)
            plt.ylim(min_flage, max_flage)
            plt.axis("equal")
            plt.grid(ls='--')
            plt.show()


gcforest = deepforest_model('./carriage.csv')
gcforest.train('first mode')



