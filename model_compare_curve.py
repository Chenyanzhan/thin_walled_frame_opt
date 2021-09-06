# -*- enconding: utf-8 -*-
# @ModuleName: model_compare_curve
# @Function:plot the curve for compare different model
# @Author: Yanzhan Chen
# @Time: 2021/6/21 20:27

import matplotlib.pyplot as plt
import numpy as np

#carriage
x_lambda = ['UK','RSM','RF','DNN','XGBoost','DF']  # x轴
# 左侧y轴：y1_recall
y1_recall = [0.3542, 0.4985, 0.5023, 0.6721, 0.5087, 0.896176]
# 右侧y轴：y2_preceise
y2_preceise = [0.6214,0.7724,0.9094,0.8291,0.9153,0.9170]
#右侧y轴: y3_preceise
y3_preceise = [0.6867,0.8672,0.9732,0.9197,0.9448,0.9616]

font_size = 13
plt.rcParams['figure.figsize'] = (7.0, 5.0)
# plt.axes().set_facecolor('whitesmoke')  # 设置背景色
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = font_size

# 绘制折线图像1, 标签，线宽
plt.plot(x_lambda, y1_recall, c='#6baf91ff',marker='*',markersize=10,label='Mass', linewidth=3)

# for a,b in zip(x_lambda,y1_recall):
#     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)


plt.plot(x_lambda, y2_preceise, c='#8bc2e9ff', marker='o',markersize=10,label='Torsional stiffness', linewidth=3)  # 同上, 'o-'
# for a,b in zip(x_lambda,y2_preceise):
#     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)


plt.plot(x_lambda,y3_preceise,c='#666666ff',marker='D',markersize=10,label ='First mode',linewidth=3)
# for a,b in zip(x_lambda,y3_preceise):
#     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)

plt.ylabel('R-square')
plt.grid(axis='x',ls='--')
plt.legend()
# plt.ylim(0.0,1.0)

# plt.grid(True,axis='y', which='major')  # 样式风格：网格型

plt.show()





# #chassis
# x_lambda = ['UK','RSM','RF','DNN','XGBoost','DF']  # x轴
# # 左侧y轴：y1_recall
# y1_recall = [0.969576, 0.957069, 0.934016, 0.985447, 0.975543, 0.985158]
# # 右侧y轴：y2_preceise
# y2_preceise = [0.948453,0.920373,0.987369,0.986564,0.986234,0.998615]
# #右侧y轴: y3_preceise
# y3_preceise = [0.974886,0.966052,0.985132,0.964220,0.966436,0.997675]
#
# font_size = 13
# plt.rcParams['figure.figsize'] = (7.0, 5.0)
# # plt.axes().set_facecolor('whitesmoke')  # 设置背景色
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.rcParams['font.size'] = font_size
#
# # 绘制折线图像1, 标签，线宽
# plt.plot(x_lambda, y1_recall, c='#6baf91ff',marker='*',markersize=10,label='Mass', linewidth=3)
#
# # for a,b in zip(x_lambda,y1_recall):
# #     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)
#
#
# plt.plot(x_lambda, y2_preceise, c='#8bc2e9ff', marker='o',markersize=10,label='Torsional stiffness', linewidth=3)  # 同上, 'o-'
# # for a,b in zip(x_lambda,y2_preceise):
# #     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)
#
#
# plt.plot(x_lambda,y3_preceise,c='#666666ff',marker='D',markersize=10,label ='Bending stiffness',linewidth=3)
# # for a,b in zip(x_lambda,y3_preceise):
# #     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)
#
# plt.ylabel('R-square')
# plt.grid(axis='x',ls='--')
# plt.legend()
# # plt.ylim(0.0,1.0)
#
# # plt.grid(True,axis='y', which='major')  # 样式风格：网格型
#
# plt.show()

# #hood
# x_lambda = ['UK','RSM','RF','DNN','XGBoost','DF']  # x轴
# # 左侧y轴：y1_recall
# y1_recall = [0.707817, 0.834774, 0.706240, 0.813522, 0.765329, 0.989037]
# # 右侧y轴：y2_preceise
# y2_preceise = [0.854585,0.869344,0.917029,0.952541,0.982273,0.984981]
# #右侧y轴: y3_preceise
# y3_preceise = [0.918059,0.928671,0.954181,0.978589,0.990894,0.992291]
#
# font_size = 13
# plt.rcParams['figure.figsize'] = (7.0, 5.0)
# # plt.axes().set_facecolor('whitesmoke')  # 设置背景色
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.rcParams['font.size'] = font_size
#
# # 绘制折线图像1, 标签，线宽
# plt.plot(x_lambda, y1_recall, c='#6baf91ff',marker='*',markersize=10,label='Mass', linewidth=3)
#
# # for a,b in zip(x_lambda,y1_recall):
# #     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)
#
#
# plt.plot(x_lambda, y2_preceise, c='#8bc2e9ff', marker='o',markersize=10,label='Torsional stiffness', linewidth=3)  # 同上, 'o-'
# # for a,b in zip(x_lambda,y2_preceise):
# #     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)
#
#
# plt.plot(x_lambda,y3_preceise,c='#666666ff',marker='D',markersize=10,label ='First mode',linewidth=3)
# # for a,b in zip(x_lambda,y3_preceise):
# #     plt.text(a,b,b,ha='center',va='top',fontsize=font_size)
#
# plt.ylabel('R-square')
# plt.grid(axis='x',ls='--')
# plt.legend()
# # plt.ylim(0.0,1.0)
#
# # plt.grid(True,axis='y', which='major')  # 样式风格：网格型
#
# plt.show()