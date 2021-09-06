# -*- enconding: utf-8 -*-
# @ModuleName: XGBoost_prediction
# @Function:
# @Author: Yanzhan Chen
# @Time: 2021/6/21 20:07
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class SVR_model():
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

        elif path == './hood.csv':
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

        data = pd.read_csv(self.path)
        X = data.iloc[:, self.start:self.end].values
        if target == 'mass':
            y = data.iloc[:, self.mass].values
        if target == 'torsion':
            y = data.iloc[:, self.torsion].values
        if target == 'mode':
            y = data.iloc[:, self.mode].values
        seed = 6
        test_size = 0.10
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state = seed)
        # model = XGBRegressor(n_estimators=19,max_depth=15)
        model = XGBRegressor()
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

support_vector_machine = SVR_model('./hood.csv')
support_vector_machine .train('mode')