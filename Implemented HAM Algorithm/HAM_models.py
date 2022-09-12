import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG, RBF, QP

import HAM_main
from HAM_main import problem_param, evaluateFunc, optimization_param, plot_param


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
            sampling = LHS(xlimits=X_test, criterion='cm', random_state=optimization_param['fix_seed'])
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
            sampling = LHS(xlimits=X_test, criterion='cm', random_state=optimization_param['fix_seed'])
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
    global Valid_Error
    global Base_Round
    global Meta_Round
    global base_model_weight
    Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
    Valid_Y = np.empty((0, 1))
    Valid_Error = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
    Base_Round = pd.DataFrame(columns=['err_m1', 'err_m2', 'err_m3', 'y_valid'])
    Meta_Round = pd.DataFrame(columns=['err_meta', 'meta_valid'])
    base_model_weight = np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)


def train_meta_model(x_sample, y_sample):
    model_krg = KRG(theta0=[1e-2], nugget=1e-3, print_global=False)
    model_rbf = RBF(d0=5, print_global=False)
    model_poly = QP(print_global=False)

    meta_model_list = [model_krg, model_rbf, model_poly]

    for meta_model in meta_model_list:
        meta_model.set_training_values(x_sample, y_sample)
        meta_model.train()

    return meta_model_list


def predict_meta_model(meta_models, x):
    meta_pred = np.zeros((x.shape[0], 1))
    for meta_model in meta_models:
        next_meta_pred = meta_model.predict_values(x)
        next_meta_pred = next_meta_pred.reshape(-1, 1)
        meta_pred = np.hstack((meta_pred, next_meta_pred))
    return meta_pred[:, 1:]


def select_n_best(x, y, n_best, minimize=True):
    y_argsort = np.argsort(y, axis=0)
    x_sort = x[y_argsort]
    y_sort = y[y_argsort]
    x_n_best = []
    y_n_best = []
    for i in range(y.shape[1]):
        if minimize:
            x_n_best.append(x_sort[:int(n_best), i, :])
            y_n_best.append(y_sort[:int(n_best), i, :])
        else:
            x_n_best.append(x_sort[-int(n_best):, i, :])
            y_n_best.append(y_sort[-int(n_best):, i, :])
    x_n_best = np.array(x_n_best)
    y_n_best = np.array(y_n_best)
    return x_n_best, y_n_best


def subspaces_generator(x_top, y_top):
    subspaces_x = [[], [], [], [], [], [], []]
    subspaces_y = [[], [], [], [], [], [], []]
    subspaces_range = []
    x_stack = np.vstack(x_top[..., :, :])
    y_stack = np.vstack(y_top[..., :, :])
    x_unique, index, count = np.unique(x_stack, axis=0, return_index=True, return_counts=True)
    if x_unique.size < x_top.size:
        for i in range(len(index)):
            if count[i] == 1:
                group = int(np.floor(index[i] / x_top.shape[1]))
                subspaces_x[group].append(x_stack[index[i]])
                subspaces_y[group].append(y_stack[index[i]])
            elif count[i] == 2:
                cnt = 0
                for j in range(x_top.shape[0]):
                    if (x_top[j, ..., :] == x_stack[index[i]]).all(1).any():
                        cnt += j
                group = np.array([[1, 2], [1, 3], [2, 3]])

                group_num = np.sum(group[cnt - 1])
                subspaces_x[group_num].append(x_stack[index[i]])
                subspaces_y[group_num].append(y_stack[index[i]])
            elif count[i] == 3:
                subspaces_x[len(subspaces_x) - 1].append(x_stack[index[i]])
                subspaces_y[len(subspaces_y) - 1].append(y_stack[index[i]])
    else:
        subspaces_x = [[x_top[0]], [x_top[1]], [x_top[2]], [], [], [], []]
        subspaces_y = [[y_top[0]], [y_top[1]], [y_top[2]], [], [], [], []]
    # reformat and find hypercube range
    for i in range(len(subspaces_x)):
        if len(subspaces_x[i]) != 0:
            subspaces_x[i] = np.array(subspaces_x[i])
            subspaces_y[i] = np.array(subspaces_y[i])
            submax = np.max(subspaces_x[i], axis=0)
            submin = np.min(subspaces_x[i], axis=0)
            subrange = np.vstack((submin, submax)).T
            subspaces_range.append(subrange)
        else:
            subrange = np.zeros((x_top.shape[2], 2))
            subspaces_range.append(subrange)
    return subspaces_x, subspaces_y, subspaces_range


def subspaces_weight(subspaces_x):
    ranges = []
    subspace_cnt = []
    for subspace in subspaces_x:
        if len(subspace) != 0:
            range_min = np.min(subspace, axis=0)
            range_max = np.max(subspace, axis=0)
            range = np.vstack((range_min, range_max))
            ranges.append(range)
            subspace_cnt.append(subspace.shape[0])
        else:
            ranges.append(np.zeros((2, problem_param['dimension'])))
            subspace_cnt.append(0)

    w = np.array(subspace_cnt)
    raw_weight = w * np.array([1, 1, 1, 2, 2, 2, 3])
    subspaces_weight = raw_weight / np.sum(raw_weight)
    return subspaces_weight


def subspaces_resample(subspaces_range, meta_models, num_samples):
    x_resample = []
    y_resample = []
    for sub_range in subspaces_range:
        sub_x_resample = HAM_main.latin_hypercube_sampling(xlimits=sub_range, num_samples=num_samples, from_problem=False)
        sub_y_resample = predict_meta_model(meta_models=meta_models, x=sub_x_resample)
        x_resample.append(sub_x_resample)
        y_resample.append(sub_y_resample)
    x_resample = np.array(x_resample)
    y_resample = np.array(y_resample)
    return x_resample, y_resample


def subspaces_selection(x, y, weight, n_select):
    select = np.rint(n_select * weight)
    point_diff = n_select * np.sum(weight) - np.sum(select)
    if point_diff != 0:
        if point_diff > 0:
            idx = np.argwhere(weight != 0).ravel()
            id_add = np.random.choice(idx, np.abs(int(np.ceil(point_diff-1e-10))))
        else:
            idx = np.argwhere(select != 0).ravel()
            id_add = np.random.choice(idx, np.abs(int(np.floor(point_diff+1e-10))))
        for k in id_add:
            select[k] += np.sign(point_diff) * 1
    if np.sum(select) != n_select:
        print('error detected')
    select_x = np.zeros((1, x.shape[2]))
    select_y = np.zeros((1, y.shape[2]))
    for i in range(len(select)):
        if len(x[i]) != 0:
            # x_n_best, y_n_best = select_n_best(x=x[i], y=y[i], n_best=select[i])
            n_best = int(select[i])
            x_unique, index, count = np.unique(x[i], axis=0, return_index=True, return_counts=True)
            id_optimal = index[np.argwhere(count > 1)].ravel()
            if n_best != 0:
                if len(id_optimal) != 0:
                    if n_best - len(id_optimal) > 0:
                        id_select_1 = id_optimal
                        id_arg = np.argsort(y[i], axis=0)
                        id_sum = np.sum(id_arg, axis=1)
                        id_sum_arg = np.argsort(id_sum, axis=0)
                        id_select_2 = np.arange(0, len(id_sum), 1)[id_sum_arg < n_best - len(id_optimal)]
                        id_select = np.append(id_select_1, id_select_2)
                    else:
                        id_select = np.random.choice(id_optimal, n_best)
                else:
                    id_arg = np.argsort(y[i], axis=0)
                    id_sum = np.sum(id_arg, axis=1)
                    id_sum_arg = np.argsort(id_sum, axis=0)
                    id_select = np.arange(0, len(id_sum), 1)[id_sum_arg < n_best]
                x_n_best = x[i, id_select.astype(int), :]
                y_n_best = y[i, id_select.astype(int), :]
                select_x = np.vstack((select_x, x_n_best))
                select_y = np.vstack((select_y, y_n_best))
            else:
                continue
        else:
            continue
    return select_x[1:], select_y[1:]


def important_region(Sample_X, traceback):
    trace_x = Sample_X[-int(traceback):, :]
    trace_max = np.max(trace_x, axis=0)
    trace_min = np.min(trace_x, axis=0)
    trace_range = np.vstack((trace_min, trace_max)).T
    return trace_range


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
                    second_layer_pred = second_layer_model.predict(xgb.DMatrix(Y_pred)).reshape(len(x1), len(x1[0]))

                    fig = plt.figure(figsize=(19.20, 10.80))
                    ax = Axes3D(fig)
                    surf_second = ax.plot_surface(x1, x2, second_layer_pred, alpha=0.5, label='dst weight', color='orange')
                    surf_real = ax.plot_surface(x1, x2, y_real, alpha=0.3, label='real', cmap=cm.coolwarm, linewidth=0,
                                                antialiased=False)
                    fig.colorbar(surf_real, shrink=0.5, aspect=5)
                    plt.show()
                    plt.close()


def errorTest(model):
    if len(model) == 3:
        if problem_param['name'] != 'chip':
            gp_test = model[0].predict_values(X_test).reshape(-1, 1)
            rbf_test = model[1].predict_values(X_test).reshape(-1, 1)
            poly_test = model[2].predict_values(X_test).reshape(-1, 1)
            Y_pred = np.hstack((gp_test, rbf_test, poly_test))
            dst_pred = np.matmul(Y_pred, base_model_weight)
            dst_error = mean_squared_error(y_test, dst_pred)
            weighted_dst_error = mean_squared_error(y_test, dst_pred, sample_weight=error_weight)
            if plot_param['error_type'] == 'weighted':
                return weighted_dst_error
            else:
                return dst_error
    elif len(model) == 4:
        if problem_param['name'] != 'chip':
            gp_test = model[0].predict_values(X_test).reshape(-1, 1)
            rbf_test = model[1].predict_values(X_test).reshape(-1, 1)
            poly_test = model[2].predict_values(X_test).reshape(-1, 1)
            Y_pred = np.hstack((gp_test, rbf_test, poly_test))
            second_layer_model = model[3]
            second_layer_pred = second_layer_model.predict(xgb.DMatrix(Y_pred)).ravel()
            second_layer_error = mean_squared_error(y_test, second_layer_pred)
            weighted_second_layer_error = mean_squared_error(y_test, second_layer_pred, sample_weight=error_weight)
            if plot_param['error_type'] == 'weighted':
                return weighted_second_layer_error
            else:
                return second_layer_error

