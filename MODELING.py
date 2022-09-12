import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, max_error, \
    median_absolute_error
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import pickle

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG, RBF, QP

from sklearn.tree import DecisionTreeRegressor

from BSSO_v3_7 import enabled_model, problem_param, evaluateFunc, Optimization_param, plot_param

base_model = [0, 0, 0, 0, 0]
second_model = []

Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
Valid_Y = np.empty((0, 1))
base_model_weight = np.array([1 / 3, 1 / 3, 1 / 3])

DST_G = []
DST_ERROR = []
SL_G = []
SL_ERROR = []


def lhsMin(num_samples):
    if problem_param['name'] != 'chip':
        global_min_pos = problem_param['global_min_pos']
        if type(problem_param['range'][0]) == list:
            X_min = problem_param['range'][0]  # 每个维度x的最小值
            X_max = problem_param['range'][1]  # 每个维度x的最大值
            X_range = (np.array(X_max) - np.array(X_min)).tolist()  # 每个维度x从最小到最大的跨度
            X_test = []  # 每个维度最优点附近的搜索域
            for i in range(problem_param['dimension']):
                X_test.append([global_min_pos[i] - X_range[i] * 1/5, global_min_pos[i] + X_range[i] * 1/5])
                if X_test[i][0] < X_min[i]:
                    X_test[i][0] = X_min[i]
                if X_test[i][1] > X_max[i]:
                    X_test[i][1] = X_max[i]
            X_test = np.array(X_test)
            sampling = LHS(xlimits=X_test, criterion='cm', random_state=Optimization_param['fix_seed'])
            x = sampling(num_samples)
        else:
            x_min = problem_param['range'][0]  # 每个维度x的最小值
            x_max = problem_param['range'][1]  # 每个维度x的最大值
            X_min = np.array(x_min).repeat(problem_param['dimension'])
            X_max = np.array(x_max).repeat(problem_param['dimension'])

            X_range = (X_max - X_min).tolist()  # 每个维度x从最小到最大的跨度
            X_test = []  # 每个维度最优点附近的搜索域
            for i in range(problem_param['dimension']):
                X_test.append([global_min_pos[i] - X_range[i] * 1 / 5, global_min_pos[i] + X_range[i] * 1 / 5])
                if X_test[i][0] < X_min[i]:
                    X_test[i][0] = X_min[i]
                if X_test[i][1] > X_max[i]:
                    X_test[i][1] = X_max[i]
            X_test = np.array(X_test)
            sampling = LHS(xlimits=X_test, criterion='cm', random_state=Optimization_param['fix_seed'])
            x = sampling(num_samples)
    else:
        x = np.zeros((1, 1))
        print('Error: optimal of chip packaging design is unknown.')
    return x

if problem_param['name'] != 'chip':
    X_test = lhsMin(problem_param['dimension'] * 500)
    y_test = evaluateFunc(X_test)
    error_weight = (y_test.max() - y_test) ** 2
    error_weight = error_weight / np.mean(error_weight)
else:
    X_test = None
    y_test = None

def paraInit():
    global Valid_Pred
    global Valid_Y
    global base_model_weight
    Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
    Valid_Y = np.empty((0, 1))
    base_model_weight = np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)


def baseModel(Sample_X, Sample_y, index):

    global base_model

    k = 5
    X = Sample_X
    y = Sample_y
    y = y.reshape(-1, 1)
    batch = (len(X) - index) / k + 1
    batch = int(batch)
    interval_start = batch * index
    interval_end = batch * (index + 1)
    X_valid = X[interval_start:interval_end, :]
    y_valid = y[interval_start:interval_end]
    X_train = np.delete(X, np.s_[interval_start:interval_end], axis=0)
    y_train = np.delete(y, np.s_[interval_start:interval_end], axis=0)

    # base model list
    model = []

    # GP base model
    if 'GP' in enabled_model:
        model_gp = KRG(theta0=[1e-2], nugget=1e-3, print_global=False)
        model_gp.set_training_values(X_train, y_train)
        model_gp.train()
        # with open('gp' + str(index) + '.model', 'wb') as file:
        #     pickle.dump(model_gp, file)
        model.append(model_gp)

    if 'RBF' in enabled_model:
        model_rbf = RBF(d0=5, print_global=False)
        model_rbf.set_training_values(X_train, y_train)
        model_rbf.train()
        a = model_rbf.predict_values(X_train)
        # with open('rbf' + str(index) + '.model', 'wb') as file:
        #     pickle.dump(model_rbf, file)
        model.append(model_rbf)

    # poly base model
    if 'Polynomial' in enabled_model:
        model_poly = QP(print_global=False)
        model_poly.set_training_values(X_train, y_train)
        model_poly.train()
        # with open('poly' + str(index) + '.model', 'wb') as file:
        #     pickle.dump(model_poly, file)
        model.append(model_poly)


    # 在validation data上进行预测，同时返回预测值和真实值
    valid_pred = []
    for i in range(len(enabled_model)):
        if enabled_model[i] == 'GP':
            pred = model_gp.predict_values(X_valid).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)
        elif enabled_model[i] == 'RBF':
            pred = model_rbf.predict_values(X_valid).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)
        else:
            pred = model_poly.predict_values(X_valid).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)

    base_model[index] = [model_gp, model_rbf, model_poly]

    return model, valid_pred, y_valid


def secondlayerModel(Valid_Pred, Valid_Y):
    global second_model

    X_second = np.hstack(Valid_Pred[0:])
    y_second = Valid_Y

    # 使用GP+采集函数作为第二层模型+选点策略
    second_model = KRG(theta0=[1e-2], nugget=1e-3, print_global=False, eval_noise=True)

    second_model.set_training_values(X_second, y_second)
    second_model.train()

    # if plot_param['train_test_error']:
    #     g_train = range(len(pred_train))
    #     plt.figure(figsize=(14.40, 9.00))
    #     plt.xlabel('Train Samples')
    #     plt.ylabel('Train Pred')
    #     plt.plot(g_train, pred_train, 'b-', lw=2)
    #     plt.plot(g_train, y_train, 'r-', lw=2)
    #     plt.show()
    #
    #     g_valid = range(len(pred_valid))
    #     plt.figure(figsize=(14.40, 9.00))
    #     plt.xlabel('Valid Samples')
    #     plt.ylabel('Valid Pred')
    #     plt.plot(g_valid, pred_valid, 'b-', lw=2)
    #     plt.plot(g_valid, y_valid, 'r-', lw=2)
    #     plt.show()
    return second_model

def corrcoef(y_test, y_pred):
    a = y_test * y_pred
    b = np.square(y_test)
    c = np.square(y_pred)
    numerator = len(y_test) * np.sum(a) - np.sum(y_test) * np.sum(y_pred)
    denominator = np.sqrt((len(y_test) * np.sum(b) - np.square(np.sum(y_test))) * (len(y_test) * np.sum(c) - np.square(np.sum(y_pred))))
    r = numerator / denominator

    if np.math.isnan(r):
        r = 1e-10
    return r

def DSTweight(Valid_Y, Valid_Pred):
    DST_MASS = np.zeros((3, 3))
    # Step-01 construct DST matrix
    for i in range(3):
        DST_MASS[i, 0] = 1 / mean_absolute_error(Valid_Y, Valid_Pred[i])
        DST_MASS[i, 1] = 1 / np.sqrt(mean_squared_error(Valid_Y, Valid_Pred[i]))
        DST_MASS[i, 2] = 1 / mean_absolute_percentage_error(Valid_Y, Valid_Pred[i])

    # Step-02 normalize DST matrix
    DST_colsum = np.sum(DST_MASS, axis=0)
    DST_MASS_TRANSFORMED = DST_MASS / DST_colsum

    # Step-03 calculate the sum of row prod
    DST_rowprod = np.prod(DST_MASS_TRANSFORMED, axis=1)
    base_model_weight = DST_rowprod / np.sum(DST_rowprod)

    return base_model_weight


# def naiveWeight(Valid_Error):
#     raw_error = np.array(Valid_Error)
#     error_sum = np.sum(raw_error, axis=1)
#     error_coef = np.max(error_sum) + np.min(error_sum) - error_sum
#     naive_weight = error_coef / np.sum(error_coef)
#     base_model_weight = naive_weight.ravel()
#     return base_model_weight


# def WTA_weight(Valid_Error):
#     raw_error = np.array(Valid_Error)
#     error_sum = np.sum(raw_error, axis=1)
#     naive_weight = (np.sum(error_sum) - error_sum) / ((np.size(error_sum) - 1) * np.sum(error_sum))
#     base_model_weight = naive_weight.ravel()
#     return base_model_weight


def modelTrain(Sample_X, Sample_y, generation):
    """
    k-fold要求初始采样点是几十一个
    根据训练种群，训练XGBoost代理模型，同时将模型保存为xgb.model文件
    :param generation: 正在进行的迭代次数
    :return: /
    """
    global Valid_Pred
    global Valid_Y
    global base_model_weight

    k = 5  # k-fold
    index = generation % k
    meta_model = None

    if index == 4:
        model, valid_pred, y_valid = baseModel(Sample_X, Sample_y, index)
        # 每五轮的矩阵拼接
        Valid_Y = np.vstack((Valid_Y, y_valid))
        for i in range(3):
            Valid_Pred[i] = np.vstack((Valid_Pred[i], valid_pred[i]))

        second_model = secondlayerModel(Valid_Pred, Valid_Y)
        model.append(second_model)
        Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
        Valid_Y = np.empty((0, 1))
        print(str(len(Sample_X)))

        # 绘制第二层模型的预测3D图
        plot3D(model)

    else:
        model, valid_pred, y_valid = baseModel(Sample_X, Sample_y, index)
        # 每五轮的矩阵拼接
        Valid_Y = np.vstack((Valid_Y, y_valid))
        for i in range(3):
            Valid_Pred[i] = np.vstack((Valid_Pred[i], valid_pred[i]))
        base_model_weight = DSTweight(Valid_Y, Valid_Pred)
        print(str(len(Sample_X)) + str(base_model_weight))

        # 绘制第二层模型的预测3D图
        plot3D(model)

    if generation == Optimization_param['generations_num'] - 1 and plot_param['error_plot']:

        plt.figure(figsize=(14.40, 9.00))
        plt.xlabel('Generations')
        plt.ylabel('Test Error')
        plt.legend("Select Points", loc='lower right')
        plt.title('Generations vs '+str(plot_param['error_type'])+' Test Error')
        plt.scatter(DST_G, DST_ERROR, alpha=1)
        plt.scatter(SL_G, SL_ERROR, alpha=1, s=80, c='r')
        plt.plot(DST_G, DST_ERROR, 'b-', lw=2)
        plt.plot(SL_G, SL_ERROR, 'r-', lw=2)
        plt.show()

        DST_G.clear()
        SL_G.clear()
        DST_ERROR.clear()
        SL_ERROR.clear()

    return base_model, meta_model, index, base_model_weight

def plot3D(model):
    if plot_param['3Dplot'] == True:
        if problem_param['dimension'] == 2:
            if type(problem_param['range'][0]) != int:
                X_min = problem_param['range'][0]  # 每个维度x的最小值
                X_max = problem_param['range'][1]  # 每个维度x的最大值
            else:
                x_min = problem_param['range'][0]  # 每个维度x的最小值
                x_max = problem_param['range'][1]  # 每个维度x的最大值
                X_min = np.array(x_min).repeat(problem_param['dimension'])
                X_max = np.array(x_max).repeat(problem_param['dimension'])
            x1 = np.linspace(X_min[0], X_max[0], 1000).reshape(-1, 1)
            x2 = np.linspace(X_min[1], X_max[1], 1000).reshape(-1, 1)

            x1, x2 = np.meshgrid(x1, x2)
            # 测试函数高度
            X1 = x1.reshape(-1, 1)
            X2 = x2.reshape(-1, 1)
            X = np.hstack((X1, X2))
            y_real = evaluateFunc(X).reshape(len(x1), len(x1[0]))

            if len(model) == 3:
                if problem_param['name'] != 'chip':
                    gp_test = model[0].predict_values(X).reshape(-1, 1)
                    rbf_test = model[1].predict_values(X).reshape(-1, 1)
                    poly_test = model[2].predict_values(X).reshape(-1, 1)
                    Y_pred = np.hstack((gp_test, rbf_test, poly_test))
                    dst_pred = np.matmul(Y_pred, base_model_weight).reshape(len(x1), len(x1[0]))

                    fig = plt.figure(figsize=(19.20, 10.80))
                    ax = Axes3D(fig)

                    surf_dst = ax.plot_surface(x1, x2, dst_pred, alpha=0.5, label='dst weight', color='orange')
                    surf_real = ax.plot_surface(x1, x2, y_real, alpha=0.3, label='real', cmap=cm.coolwarm, linewidth=0, antialiased=False)
                    fig.colorbar(surf_real, shrink=0.5, aspect=5)
                    plt.show()
                    plt.close()

            elif len(model) == 4:
                if problem_param['name'] != 'chip':
                    gp_test = model[0].predict_values(X).reshape(-1, 1)
                    rbf_test = model[1].predict_values(X).reshape(-1, 1)
                    poly_test = model[2].predict_values(X).reshape(-1, 1)
                    Y_pred = np.hstack((gp_test, rbf_test, poly_test))
                    second_layer_model = model[3]
                    second_layer_pred = second_layer_model.predict_values(Y_pred).reshape(len(x1), len(x1[0]))

                    fig = plt.figure(figsize=(19.20, 10.80))
                    ax = Axes3D(fig)
                    surf_second = ax.plot_surface(x1, x2, second_layer_pred, alpha=0.5, label='dst weight', color='orange')
                    surf_real = ax.plot_surface(x1, x2, y_real, alpha=0.3, label='real', cmap=cm.coolwarm, linewidth=0,
                                                antialiased=False)
                    fig.colorbar(surf_real, shrink=0.5, aspect=5)
                    plt.show()
                    plt.close()

# def errorTest(model):
#     if len(model) == 3:
#         if problem_param['name'] != 'chip':
#             gp_test = model[0].predict_values(X_test).reshape(-1, 1)
#             rbf_test = model[1].predict_values(X_test).reshape(-1, 1)
#             poly_test = model[2].predict_values(X_test).reshape(-1, 1)
#             Y_pred = np.hstack((gp_test, rbf_test, poly_test))
#             dst_pred = np.matmul(Y_pred, base_model_weight)
#             dst_error = mean_squared_error(y_test, dst_pred)
#             weighted_dst_error = mean_squared_error(y_test, dst_pred, sample_weight=error_weight)
#             if plot_param['error_type'] == 'weighted':
#                 return weighted_dst_error
#             else:
#                 return dst_error
#     elif len(model) == 4:
#         if problem_param['name'] != 'chip':
#             gp_test = model[0].predict_values(X_test).reshape(-1, 1)
#             rbf_test = model[1].predict_values(X_test).reshape(-1, 1)
#             poly_test = model[2].predict_values(X_test).reshape(-1, 1)
#             Y_pred = np.hstack((gp_test, rbf_test, poly_test))
#             second_layer_model = model[3]
#             second_layer_pred = second_layer_model.predict_values(Y_pred).ravel()
#             second_layer_error = mean_squared_error(y_test, second_layer_pred)
#             weighted_second_layer_error = mean_squared_error(y_test, second_layer_pred, sample_weight=error_weight)
#             if plot_param['error_type'] == 'weighted':
#                 return weighted_second_layer_error
#             else:
#                 return second_layer_error

