import os
import pickle
import warnings
import random

import matplotlib
import numpy as np
import pandas as pd
import scipy
import xgboost as xgb
from matplotlib import patches
from scipy.special import erfc
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_squared_error
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt

# import其他py文件
from pyswarms.single import GlobalBestPSO

import MODELING as mt

# Base model settings
enabled_model = np.array([
    'GP',# KRG
    'RBF',
    'Polynomial',# PRS
])

# Test problem settings
problem_param = {
    # 'name': 'rosenbrock',
    # 'dimension': 10,
    # 'range': [-2.048, 2.048],
    # 'global_min_pos': [1] * 10,
    # 'min': 0,

    # 'name': 'rastrigin',
    # 'dimension': 10,
    # 'range': [-5, 5],
    # 'global_min_pos': [0] * 10,
    # 'min': 0,

    # 'name': 'griewank',
    # 'dimension': 20,
    # 'range': [-600, 600],
    # 'global_min_pos': [0] * 20,
    # 'min': 0,

    # 'name': 'ellipsoid',
    # 'dimension': 10,
    # 'range': [-5.12, 5.12],
    # 'global_min_pos': [0] * 10,
    # 'min': 0,

    # 'name': 'ackley',
    # 'dimension': 20,
    # 'range': [-32, 32],
    # 'global_min_pos': [0] * 20,
    # 'min': 0,

    # 'name': 'shcb',
    # 'dimension': 2,
    # 'range': [[-3, -2], [3, 2]],
    # 'min': -1.0316,
    # 'global_min_pos': [0.0898, -0.7126],
    # or [-0.0898, 0.7126]

    # 'name': 'goldstein_price',
    # 'dimension': 2,
    # 'range': [-2, 2],
    # 'min': 3,
    # 'global_min_pos': [0, -1],

    # 'name': 'hartman3',
    # 'dimension': 3,
    # 'range': [0, 1],
    # 'min': -3.86278,
    # 'global_min_pos': [0.114614, 0.555649, 0.852547],

    # 'name': 'alpine',
    # 'dimension': 2,
    # 'range': [-10, 10],
    # 'min': 0,
    # 'global_min_pos': [0]*2,

    # 'name': 'hartman6',
    # 'dimension': 6,
    # 'range': [0, 1],
    # 'min': -3.32237,
    # 'global_min_pos': [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],

    # 'name': 'easom',
    # 'dimension': 2,
    # 'range': [-10, 10],
    # 'min': -1,
    # 'global_min_pos': [np.pi, np.pi],

    # 'name': 'shekel',
    # 'dimension': 4,
    # 'range': [0, 10],
    # 'min': -10.1532,
    # 'global_min_pos': [4, 4, 4, 4],

    # 'name': 'eggholder',
    # 'dimension': 2,
    # 'range': [-512, 512],
    # 'min': -959.6407,
    # 'global_min_pos': [512, 404.2319],

    # 'name': 'branin',
    # 'dimension': 2,
    # 'range': [[-5, 0], [10, 15]],
    # 'min': 0.397887,
    # 'global_min_pos': [9.42478, 2.475],
    # or [-np.pi, 12.275], [np.pi, 2.275]

    # 'name': 'chip',
    # 'dimension': 5,
    # 'range': [[0.55, 0.2, 0.2, 0.02, 8], [0.95, 0.3, 0.32, 0.04, 12]],
    # 'min': 0,
}

Optimization_param = {
    # 'sample_init_num': 20,
    # 'generations_num': 80,
    'sample_init_num': 5 * problem_param['dimension'],
    'generations_num': 11 * problem_param['dimension'],
    'runs_num': 2,
    'init_seed': [1, 2],
    'fix_seed': int(np.random.rand(1) * 1e3),
    'current_generation': 0,
}

# EA Algorithm settings
PSO_param = {
    'n_particles': 150 * problem_param['dimension'],
    'local_iters': 2000,
    'global_iters': 2000,
    'local_search': True,
    'local_search_num': int(problem_param['dimension'] * 1.5), # 1.5 3
    'local_shrink_scale': 1,
    'local_mode': 'optimal',  # 'nearest', 'optimal'
    'global_clusters': 3,
    'explore': False,
    'explore_prob_offset': 0.7,
    'pso_options': {'c1': 2.05, 'c2': 2.05, 'w': 0.7},
    'pso_global_options': {'c1': 2.05, 'c2': 2.05, 'w': 0.7},
    'display': False,
}

plot_param = {
    'train_test_error': False,
    'region_plot': False,
    'error_plot': False,
    'sample_plot': False,
    'result_plot': True,
    'cluster_plot': False,
    '3Dplot': False,
    'error_type': 'weighted',  # 'weighted', 'normal'
    # 'contour_range': 'None',  # 'None', 'global', 'local'
    'contour_range': 'global',
    # 'contour_range': 'local',
}

base_model_weight = np.ones(len(enabled_model)).ravel() / len(enabled_model)

def evaluateFunc(sample_array):
    """
    Expensive optimization of test functions and chip packaging problem
    :param sample_array: sample points; 2D numpy array
    :return X: the same as sample_array
    :return y: corresponding values
    """
    X = sample_array
    result = None
    if problem_param['name'] == 'rosenbrock':
        result = np.sum(100 * np.square(X[:, 1:] - np.square(X[:, :-1])) + np.square(X[:, -1] - 1).reshape(-1, 1),
                        axis=1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        elif problem_param['dimension'] == 2:
            problem_param['column_name'] = ['x1', 'x2', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'rastrigin':
        result = 10 * problem_param['dimension'] + np.sum(np.square(X) - 10 * np.cos(2 * np.pi * X), axis=1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'griewank':
        den = 1 / np.sqrt(np.arange(1, problem_param['dimension'] + 1))
        result = np.sum(np.square(X), axis=1) / 4e3 - np.prod(np.cos(np.multiply(X, den)), axis=1) + 1
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'ellipsoid':
        i = np.arange(1, problem_param['dimension'] + 1)
        result = np.sum(np.square(X) * i, axis=1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'goldstein_price':
        x1, x2 = X[:, 0], X[:, 1]
        result_a = 1 + (np.square(x1 + x2 + 1)) * \
                   (19 - 14 * x1 + 3 * np.square(x1)
                    - 14 * x2 + 6 * x1 * x2 + 3 * np.square(x2))
        result_b = 30 + (np.square(2 * x1 - 3 * x2)) * \
                   (18 - 32 * x1 + 12 * np.square(x1)
                    + 48 * x2 - 36 * x1 * x2 + 27 * np.square(x2))
        result = result_a * result_b
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'hartman3':
        ALPHA = np.array([[1.0], [1.2], [3.0], [3.2]])
        A = np.array([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]]).repeat(X.shape[0], axis=0)
        P = 0.0001 * np.array([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]]).repeat(X.shape[0], axis=0)
        X_trans = np.tile(X, (4, 1))
        inner_sum = np.sum(np.multiply(A, np.square(X_trans - P)), axis=1).reshape(4, -1)
        result = - np.sum(ALPHA * np.exp(-inner_sum), axis=0)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'y']
    elif problem_param['name'] == 'hartman6':
        ALPHA = np.array([[1.0], [1.2], [3.0], [3.2]])
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]).repeat(X.shape[0], axis=0)
        P = 0.0001 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]).repeat(X.shape[0], axis=0)
        X_trans = np.tile(X, (4, 1))
        inner_sum = np.sum(np.multiply(A, np.square(X_trans - P)), axis=1).reshape(4, -1)
        result = - np.sum(ALPHA * np.exp(-inner_sum), axis=0)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']
    elif problem_param['name'] == 'ackley':
        result = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(X), axis=1) / problem_param['dimension'])) - np.exp(
            np.sum(np.cos(2 * np.pi * X), axis=1) / problem_param['dimension']) + 20 + np.exp(1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        elif problem_param['dimension'] == 2:
            problem_param['column_name'] = ['x1', 'x2', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'shcb':
        x1, x2 = X[:, 0], X[:, 1]
        a = x1 * x2
        result = (4 - 2.1 * np.square(x1) + np.power(x1, 4) / 3) * np.square(x1) + x1 * x2 + (
                -4 + 4 * np.square(x2)) * np.square(x2)
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'easom':
        x1, x2 = X[:, 0], X[:, 1]
        result = -np.cos(x1) * np.cos(x2) * np.exp(-np.square(x1 - np.pi) - np.square(x2 - np.pi))
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'alpine':
        if problem_param['dimension'] == 2:
            x1, x2 = X[:, 0], X[:, 1]
            result = np.abs(x1 * np.sin(x1) + 0.1 * x1) + np.abs(x2 * np.sin(x2) + 0.1 * x2)
            problem_param['column_name'] = ['x1', 'x2', 'y']
        elif problem_param['dimension'] == 5:
            result = np.sum(np.abs(X * np.sin(X) + 0.1 * X), axis=1)
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
    elif problem_param['name'] == 'shekel':
        m = 5
        C = np.array([
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
        ])
        beta = 0.1 * np.array([[1, 2, 2, 4, 4, 6, 3, 7, 5, 5]])
        C = np.tile(C[:, :m].T, (X.shape[0], 1))
        X_trans = X.repeat(m, axis=0)
        result_p = np.sum(np.square(X_trans - C), axis=1).reshape(-1, m) + beta[0, :m]
        result = - np.sum(1 / result_p, axis=1)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'y']
    elif problem_param['name'] == 'eggholder':
        x1, x2 = X[:, 0], X[:, 1]
        result = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - x2 - 47)))
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'branin':
        x1, x2 = X[:, 0], X[:, 1]
        a = 1
        b = 5.1 / (4 * np.square(np.pi))
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        result = a * np.square(x2 - b * np.square(x1) + c * x1 - r) + s * (1 - t) * np.cos(x1) + s
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'chip':
        result = warpSimulation(sample_array)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
    else:
        print("New test function")
    y = result.reshape(-1, 1)
    return y

def warpSimulation(design_para):
    # 输入要求为一个二维向量，本方法会自动展为一维向量
    design_para = design_para.ravel()
    design_csv = np.savetxt('chip\\ABAQUS_Simulation\\design_para.csv', design_para)
    cmd_cae = os.system('abaqus cae noGUI=chip\\ABAQUS_Simulation\\chip_simulation.py')
    root_path = os.getcwd()
    with open(root_path + '\\chip\\ABAQUS_Simulation\\chip_warpage.csv') as f:
        warpageData = np.loadtxt(f)
    warpage = np.abs(warpageData[0])
    return warpage


# Latin Hypercube Sampling
def latin_hypercube_sampling(num_samples):
    if type(problem_param['range'][0]) == list:
        X_min = problem_param['range'][0]
        X_max = problem_param['range'][1]
        X_range = []
        for i in range(problem_param['dimension']):
            X_range.append([X_min[i], X_max[i]])
        X_range = np.array(X_range)
        sampling = LHS(xlimits=X_range, criterion='cm', random_state=Optimization_param['init_seed'][run])
        x = sampling(num_samples)
    else:
        x_lim = np.array(problem_param['range']).reshape(-1, 1)
        x_lim = x_lim.repeat(problem_param['dimension'], axis=1).T
        sampling = LHS(xlimits=x_lim, criterion='cm', random_state=Optimization_param['init_seed'][run])
        x = sampling(num_samples)
    return x


def decay(y_scale, x_offset, y_offset):
    x = Optimization_param['current_generation']
    total_x = Optimization_param['generations_num']
    order = 0.1
    y = 1 - 1.0 / (1 + np.exp(-(x - total_x * x_offset)) ** order)
    y = y * y_scale + y_offset
    y = np.clip(y, a_min=0, a_max=1)
    return int(np.floor(y))


def globalCluster(pso_global_pos, pso_global_cost):
    # Kmeans 聚类
    # n_clusters = PSO_param['global_clusters']
    # select_pos = []
    # select_cost = []
    # input = pso_global_pos
    # kms = KMeans(init='k-means++', n_clusters=n_clusters, random_state=Optimization_param['fix_seed'], tol=1e-4)
    # cluster_index = kms.fit_predict(input)
    # cluster_samples = []
    # for i in range(n_clusters):
    #     if np.any(cluster_index[0:] == i):
    #         cluster_pos = pso_global_pos[cluster_index[0:] == i, :]
    #         cluster_cost = pso_global_cost[cluster_index[0:] == i].reshape(-1, 1)
    #         cluster_samples.append(cluster_pos)
    #         cluster_array = np.hstack((cluster_pos, cluster_cost))
    #         cluster_array = cluster_array[np.argsort(cluster_array[:, -1]), :]
    #         select_pos.append(cluster_array[0, :-1])
    #         select_cost.append(cluster_array[0, -1])

    # DBSCAN 聚类
    select_pos = []
    select_cost = []
    input = pso_global_pos
    attempt = 3
    cluster_index = 0
    for i in range(attempt):
        coeff = 1 / 30
        if type(problem_param['range'][0]) == list:
            length = problem_param['range'][1][0] - problem_param['range'][0][0]
            width = problem_param['range'][1][1] - problem_param['range'][0][1]
            eps = np.sqrt(np.square(length) + np.square(width)) * coeff

        else:
            length = problem_param['range'][1] - problem_param['range'][0]
            eps = np.sqrt(problem_param['dimension'] * np.square(length)) * coeff
        eps = eps * (i + 1)
        cluster = DBSCAN(eps=eps, min_samples=5)
        cluster_index = cluster.fit_predict(input)
        if 4 * len(cluster_index[cluster_index < 0]) < len(cluster_index):
            break
        elif i == attempt - 1:
            attempt = -1

    cluster_samples = []
    for i in range(np.min(cluster_index), np.max(cluster_index) + 1):
        # if i != -1:
        if np.any(cluster_index[0:] == i):
            cluster_pos = pso_global_pos[cluster_index[0:] == i, :]
            cluster_cost = pso_global_cost[cluster_index[0:] == i].reshape(-1, 1)
            cluster_samples.append(cluster_pos)
            cluster_array = np.hstack((cluster_pos, cluster_cost))
            cluster_array = cluster_array[np.argsort(cluster_array[:, -1]), :]
            select_pos.append(cluster_array[0, :-1])
            select_cost.append(cluster_array[0, -1])

    samplesContour(contour_range=plot_param['contour_range'], region_plot=False,
                   region_param=None, samples_plot=False, Sample_X_init=None, Sample_Select_X=None,
                   cluster_plot=plot_param['cluster_plot'], cluster_samples=cluster_samples, select_pos=select_pos)

    choice = int(np.random.randint(0, len(select_pos), 1))
    pso_cluster_pos = np.array(select_pos[choice])
    pso_cluster_cost = np.array(select_cost[choice])
    return pso_cluster_pos.reshape(1, -1), pso_cluster_cost


def explore_prob():
    if PSO_param['explore']:
        gen = int(Optimization_param['current_generation'] / Optimization_param['generations_num'] * 100)
        rand = np.random.rand()
        order = 0.15
        scale = 0.2
        x_offset = 0.5
        y_offset = PSO_param['explore_prob_offset']
        y = 1 - 1.0 / (1 + np.exp(-(gen - 100 * x_offset)) ** order)
        y = y * scale + y_offset
        return rand > y
    else:
        return False


def explore(sample_point, model_num, base_model_weight):
    dim = problem_param['dimension']
    xl = problem_param['range'][0]
    xu = problem_param['range'][1]
    repeat = 1e4
    explore_dim = np.random.randint(0, dim, 1)[0]
    sequence = np.arange(xl, xu, 1 / repeat)
    explore_mat = sample_point.reshape(-1, 1).T.repeat((xu - xl) * repeat, axis=0)
    explore_mat[:, explore_dim] = sequence
    explore_eval = popEvaluate(explore_mat, model_num, base_model_weight)
    explore_optimal_point = explore_mat[np.argmin(explore_eval), :]
    explore_optimal_cost = explore_eval.min()
    return explore_optimal_point.reshape(1, -1), explore_optimal_cost

def EIacquisition(mu, std, fMin, epsilon):
    """
    Expected improvement acquisition function
    INPUT:
        - muNew: mean of predicted point in grid
        - stdNew: sigma (square root of variance) of predicted point in grid
        - fMax: observed or predicted maximum value (depending on noise p.19 Brochu et al. 2010)
        - epsilon: trade-off parameter (>=0)
            [Lizotte 2008] suggest setting epsilon = 0.01 (scaled by the signal variance if necessary)  (p.14 [Brochu et al. 2010])
    OUTPUT:
        - EI: expected improvement for candidate point
    As describend in:
        E Brochu, VM Cora, & N de Freitas (2010):
        A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning,
        arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
    """

    if isinstance(std, np.ndarray):
        std[std<1e-10] = 1e-10
    elif std< 1e-10:
        std = 1e-10

    u = (fMin - mu - epsilon)/std
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))

    f_acqu = std * (u * Phi + phi)
    return f_acqu


def popEvaluate(population, iter_index, base_model_weight, Sample_y):

    if iter_index == 4:
        pop_array = np.vstack((population[0:]))
        Pred_gp = np.empty((len(population), 1))
        Pred_rbf = np.empty((len(population), 1))
        Pred_poly = np.empty((len(population), 1))

        for i in range(5):
            # gp/krg base model
            model_gp = mt.base_model[i][0]
            gp_pred = model_gp.predict_values(pop_array).reshape(-1, 1)

            # rbf base model
            model_rbf = mt.base_model[i][1]
            rbf_pred = model_rbf.predict_values(pop_array).reshape(-1, 1)

            # poly base model
            model_poly = mt.base_model[i][2]
            poly_pred = model_poly.predict_values(pop_array).reshape(-1, 1)

            Pred_gp = np.hstack((Pred_gp, gp_pred))
            Pred_rbf = np.hstack((Pred_rbf, rbf_pred.reshape(-1, 1)))
            Pred_poly = np.hstack((Pred_poly, poly_pred))

        Pred_gp = Pred_gp[:, 1:]
        Pred_rbf = Pred_rbf[:, 1:]
        Pred_poly = Pred_poly[:, 1:]

        Pred_gp = np.mean(Pred_gp, axis=1).reshape(-1, 1)
        Pred_rbf = np.mean(Pred_rbf, axis=1).reshape(-1, 1)
        Pred_poly = np.mean(Pred_poly, axis=1).reshape(-1, 1)

        # second-layer model here
        second_input = np.hstack((Pred_gp, Pred_rbf, Pred_poly))

        second_model = mt.second_model

        y_mean = second_model.predict_values(second_input).ravel()
        y_var = second_model.predict_variances(second_input).ravel()

        y_std = np.sqrt(y_var)

        # fMin = min(Sample_y)
        #
        # EI = EIacquisition(y_mean, y_std, fMin, 0.01)
        # a = 1

        return y_mean

    else:
        pop_array = np.vstack((population[0:]))

        # gp metamodel
        model_gp = mt.base_model[iter_index][0]
        gp_pred = model_gp.predict_values(pop_array)
        pop_pred_gp = gp_pred.reshape(-1, 1)

        # rbf metamodel
        model_rbf = mt.base_model[iter_index][1]
        rbf_pred = model_rbf.predict_values(pop_array)
        pop_pred_rbf = rbf_pred.reshape(-1, 1)

        # poly base model
        model_poly = mt.base_model[iter_index][2]
        poly_pred = model_poly.predict_values(pop_array)
        pop_pred_poly = poly_pred.reshape(-1, 1)

        # 各base model赋权重
        pop_pred = base_model_weight[0] * pop_pred_gp + base_model_weight[1] * pop_pred_rbf + base_model_weight[2] * pop_pred_poly
        pop_pred = pop_pred.ravel()

        return pop_pred


Region_Param = []
def localRegion(Sample_X, Sample_y, bounds, mode='optimal'):
    k_fold = 5
    num_local_search = PSO_param['local_search_num']

    exact_sample = Sample_X
    exact_value = Sample_y
    exact_sample = exact_sample[np.argsort(exact_value.ravel()), :]
    dist = exact_sample - exact_sample[0, :].reshape(1, -1)
    dist = np.linalg.norm(dist, axis=1)
    dist_index = np.argsort(dist)
    dist_optimal = exact_sample[dist_index]
    if mode == 'nearest':
        near_optimal = dist_optimal[1:1 + num_local_search, :]
    elif mode == 'optimal':
        near_optimal = exact_sample[0:num_local_search]
    else:
        near_optimal = None

    range_max = np.max(near_optimal, axis=0)
    range_min = np.min(near_optimal, axis=0)
    range_mean = np.mean(near_optimal, axis=0)
    range_lb = (range_min * int(100 * PSO_param['local_shrink_scale']) + range_mean * int(
        100 * (1 - PSO_param['local_shrink_scale']))) / 100
    range_ub = (range_max * int(100 * PSO_param['local_shrink_scale']) + range_mean * int(
        100 * (1 - PSO_param['local_shrink_scale']))) / 100
    updated_bounds = (range_lb, range_ub)

    if problem_param['dimension'] == 2:
        x1_lb = updated_bounds[0][0]
        x2_lb = updated_bounds[0][1]
        x1_ub = updated_bounds[1][0]
        x2_ub = updated_bounds[1][1]
        x1_len = x1_ub - x1_lb
        x2_len = x2_ub - x2_lb

        # 绘制重点区域
        xy = (x1_lb, x2_lb)
        width = x1_len
        height = x2_len
        region_param = [xy, width, height]
        Region_Param.append(region_param)

        if Optimization_param['current_generation'] == Optimization_param['generations_num'] - 1:
            samplesContour(contour_range=plot_param['contour_range'], region_plot=plot_param['region_plot'],
                           region_param=Region_Param, samples_plot=False, Sample_X_init=None, Sample_Select_X=None,
                           cluster_plot=False, cluster_samples=None, select_pos=None)
            Region_Param.clear()
    return updated_bounds

def PSO(Sample_X, Sample_y, model_num, base_model_weight):
    dim = problem_param['dimension']
    x_min = problem_param['range'][0] * np.ones(dim)
    x_max = problem_param['range'][1] * np.ones(dim)
    global_bounds = (x_min, x_max)
    # PSO参数设置
    iter_index = Optimization_param['current_generation'] % 5
    # local search
    if iter_index == 4 or iter_index == 3 or iter_index == 2:
        local_bounds = localRegion(Sample_X, Sample_y, global_bounds, mode=PSO_param['local_mode'])
        optimizer = GlobalBestPSO(
            n_particles=PSO_param['n_particles'],
            dimensions=dim,
            options=PSO_param['pso_options'],
            bounds=local_bounds, )
    else:
        c2 = PSO_param['pso_global_options']['c2'] / Optimization_param['generations_num'] * (Optimization_param['current_generation'] + 1)
        options = {'c1': PSO_param['pso_global_options']['c1'], 'c2': c2, 'w': PSO_param['pso_global_options']['w']}
        optimizer = GlobalBestPSO(
            n_particles=PSO_param['n_particles'],
            dimensions=dim,
            options=options,
            bounds=global_bounds,)


    cost, pos = optimizer.optimize(
        popEvaluate,
        iters=PSO_param['local_iters'],
        iter_index=model_num,
        base_model_weight=base_model_weight,
        Sample_y=Sample_y,
        verbose=PSO_param['display'], )

    cost = np.array(cost).reshape(1, -1)
    pos = pos.reshape(1, -1)

    # 局部搜索：选择最优点
    prob = explore_prob()
    if prob or (not PSO_param['explore']):
        if iter_index == 4 or iter_index == 3 or iter_index == 2:
            pso_pos = pos
            pso_cost = cost
        else:
            pso_global_pos = optimizer.swarm.pbest_pos
            pso_global_cost = optimizer.swarm.pbest_cost
            pso_pos, pso_cost = globalCluster(pso_global_pos, pso_global_cost)
    else:
        pso_pos, pso_cost = explore(pos, model_num, base_model_weight)

    return pso_pos, pso_cost.reshape(-1, 1)

def SAiterate():
    global Optimization_param
    mt.paraInit()
    if problem_param['name'] == 'chip':
        if Optimization_param['sample_init_num'] == 20:
            with open('chip\\ABAQUS_Simulation\\chip_init_data_20d.csv') as f:
                init_data = pd.read_csv(f, header=0, index_col=0)
            Sample_X_init = np.array(init_data.iloc[:, :-1])
            Sample_y_init = np.array(init_data.iloc[:, -1]).reshape(-1, 1)
        else:
            Sample_X_init = latin_hypercube_sampling(Optimization_param['sample_init_num'])
            Sample_y_init = []
            for sample_x in Sample_X_init:
                sample_y = warpSimulation(sample_x)
                Sample_y_init.append(sample_y)
            Sample_y_init = np.array(Sample_y_init).reshape(-1, 1)
            Sample_Array = np.hstack((Sample_X_init, Sample_y_init))
            init_data = pd.DataFrame(data=Sample_Array, columns=problem_param['column_name'])

            root_path = os.getcwd()
            init_data.to_csv(root_path + '\\chip\\ABAQUS_Simulation\\' + 'chip_init_data_' + str(
                problem_param['dimension']) + 'd.csv', encoding='gbk')
    else:
        Sample_X_init = latin_hypercube_sampling(Optimization_param['sample_init_num'])
        Sample_y_init = evaluateFunc(Sample_X_init)

    Sample_X = Sample_X_init
    Sample_y = Sample_y_init
    Sample_Select_X = []
    Sample_Select_y = []
    for generation in range(0, Optimization_param['generations_num']):
        Optimization_param['current_generation'] = generation
        # 采样点训练代理模型
        base_model, meta_model, model_num, base_model_weight = mt.modelTrain(Sample_X, Sample_y, generation)
        # PSO
        sample_x, _ = PSO(Sample_X, Sample_y, model_num, base_model_weight)

        # 仿真评估所选点
        sample_y = evaluateFunc(sample_x)
        Sample_X = np.append(Sample_X, sample_x, axis=0)
        Sample_y = np.append(Sample_y, sample_y, axis=0)
        Sample_Select_X.append(sample_x)
        Sample_Select_y.append(sample_y)

        print('X:' + str(sample_x[0]) + 'y:' + str(sample_y[0]))


    Sample_Select_X = np.vstack((np.array(Sample_Select_X)[0:]))
    Sample_Select_y = np.array(Sample_Select_y).reshape(-1, 1)
    mt.paraInit()
    # 绘制收敛曲线图
    resultDisp(Sample_Select_y, Sample_y_init)
    # 绘制等高线及选点图
    samplesContour(contour_range=plot_param['contour_range'], region_plot=False, region_param=None,
                   samples_plot=plot_param['sample_plot'],
                   Sample_X_init=Sample_X_init, Sample_Select_X=Sample_Select_X, cluster_plot=False,
                   cluster_samples=None, select_pos=None)

    # 输出相关数据
    Sample_Array = np.hstack((Sample_X, Sample_y))
    Sample_Array = pd.DataFrame(Sample_Array, columns=problem_param['column_name'])
    Sample_Array = Sample_Array.sort_values(by='y', axis=0, ascending=True)
    Sample_Optimum = np.array(Sample_Array.iloc[0, :])
    root_path = os.getcwd()
    name = root_path + '\\results\\samples\\' + str(problem_param['name']) + '\\'\
        + str(Optimization_param['init_seed'][run]) + '_'\
        + str(Optimization_param['sample_init_num']) + '+'\
        + str(Optimization_param['generations_num']) + '_samples_' \
        + str(problem_param['name']) + '_' \
        + str(problem_param['dimension']) + 'd' + '.csv'
    if not os.path.exists(root_path + '\\results\\samples\\' + str(problem_param['name'])):
        os.makedirs(root_path + '\\results\\samples\\' + str(problem_param['name']))
    Sample_Array.to_csv(name, encoding='gbk')
    return Sample_Array, Sample_Optimum


def samplesContour(contour_range, region_plot, region_param, samples_plot, Sample_X_init, Sample_Select_X, cluster_plot, cluster_samples, select_pos):
    if problem_param['dimension'] == 2 and (region_plot or samples_plot or cluster_plot):
        if contour_range == 'global':
            # 全局画图
            if type(problem_param['range'][0]) == list:
                X_min = problem_param['range'][0]  # 每个维度x的最小值
                X_max = problem_param['range'][1]  # 每个维度x的最大值

            else:
                x_min = problem_param['range'][0]  # 每个维度x的最小值
                x_max = problem_param['range'][1]  # 每个维度x的最大值
                X_min = np.array(x_min).repeat(problem_param['dimension'])
                X_max = np.array(x_max).repeat(problem_param['dimension'])
            x1 = np.linspace(X_min[0], X_max[0], 1000)
            x2 = np.linspace(X_min[1], X_max[1], 1000)
        elif contour_range == 'local':
            # 最优局部画图
            global_min_pos = problem_param['global_min_pos']
            if type(problem_param['range'][0]) == list:
                X_min = problem_param['range'][0]  # 每个维度x的最小值
                X_max = problem_param['range'][1]  # 每个维度x的最大值
                X_range = (np.array(X_max) - np.array(X_min)).tolist()  # 每个维度x从最小到最大的跨度
                X_test = []  # 每个维度最优点附近的搜索域
                for i in range(problem_param['dimension']):
                    X_test.append([global_min_pos[i] - X_range[i] * 1 / 5, global_min_pos[i] + X_range[i] * 1 / 5])
                    if X_test[i][0] < X_min[i]:
                        X_test[i][0] = X_min[i]
                    if X_test[i][1] > X_max[i]:
                        X_test[i][1] = X_max[i]
                X_test = np.array(X_test)
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
            x1 = np.linspace(X_test[0][0], X_test[0][1], 1000)
            x2 = np.linspace(X_test[1][0], X_test[1][1], 1000)
        if contour_range != 'None':
            x1, x2 = np.meshgrid(x1, x2)
            # 测试函数高度
            X1 = x1.reshape(-1, 1)
            X2 = x2.reshape(-1, 1)
            X = np.hstack((X1, X2))
            z = evaluateFunc(X).reshape(len(x1), len(x1[0]))

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            contour = ax.contour(x1, x2, z, 30, colors="black", linewidths=0.5)
            plt.clabel(contour, fontsize=6, inline=True)
            ax.contourf(x1, x2, z, 30, cmap="afmhot_r", alpha=0.5)
            # "afmhot_r", "BrBG"
            # 添加colorbar
            fig.colorbar(contour, ax=ax)
            ax.set_title(problem_param['name'] + " contour")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            if contour_range == 'local':
                ax.set_xlim((X_test[0][0], X_test[0][1]))
                ax.set_ylim((X_test[1][0], X_test[1][1]))

            # 绘制重点局部区域变化：
            if region_plot == True:
                index = np.arange(1, len(region_param) + 1)
                for i, region in zip(range(len(region_param)), region_param):
                    ax.add_patch(patches.Rectangle(region[0], region[1], region[2], edgecolor='red', facecolor='green',
                                                   fill=False))
                    ax.annotate(index[i], xy=(region[0][0], region[0][1]))
                plt.show()

            # 绘制初始采样点及后续选点：
            if samples_plot == True:
                # 绘制初始采样点
                # x1_init = Sample_X_init[:, 0].ravel()
                # x2_init = Sample_X_init[:, 1].ravel()
                # plt.scatter(x1_init, x2_init, s=20, c='m')
                # 绘制后续迭代选点
                x1_select = Sample_Select_X[:, 0].ravel()
                x2_select = Sample_Select_X[:, 1].ravel()
                index = np.arange(1, len(Sample_Select_X) + 1)

                color_values = index * 0.5
                cm = plt.cm.get_cmap('Blues')

                for i in range(len(x1_select)):
                    ax.annotate(index[i], xy=(x1_select[i], x2_select[i]))
                    if i % 5 == 4:
                        ax.scatter(x1_select[i], x2_select[i], s=20, c='r', marker='*')
                    else:
                        ax.scatter(x1_select[i], x2_select[i], s=20, c='b', marker='*')
                        # plt.scatter(x1_select[i], x2_select[i], s=20, c=color_values[i], marker='*', cmap=cm)

                plt.show()

            if cluster_plot == True:
                colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
                for cluster, i in zip(cluster_samples, range(len(cluster_samples))):
                    ax.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], cmap='plasma')
                    ax.scatter(select_pos[i][0], select_pos[i][1], c='m', marker='*', s=100)

                plt.show()


def resultDisp(Sample_Select_y, Sample_y_init):
    Min = []  # 每一代时所有采样点的最小值
    min = np.min(Sample_y_init)  # 截止该次代时的最优解

    for i in range(0, len(Sample_Select_y)):
        if Sample_Select_y[i] < min:
            Min.append(float(Sample_Select_y[i]))
            min = float(Sample_Select_y[i])
        else:
            Min.append(min)

    meta_values = []
    for index in range(len(Sample_Select_y)):
        if index % 5 == 4:
            meta_values.append(Sample_Select_y[index])
    meta_values = np.array(meta_values).reshape(-1, 1)

    if plot_param['result_plot']:
        g = range(Optimization_param['generations_num'])
        meta_index = np.arange(4, Optimization_param['generations_num'], 5).reshape(-1, 1)

        plt.figure(figsize=(14.40, 9.00))
        plt.xlabel('Generations')
        plt.ylabel('Test Values')
        plt.legend("Select Points", loc='lower right')
        plt.title('Generations vs TestValues')
        plt.plot(g, Sample_Select_y, 'r-', lw=1)
        plt.scatter(g, Sample_Select_y, alpha=1)
        plt.scatter(meta_index, meta_values, alpha=1, s=80, c='r')

        plt.legend("Convergence Curve", loc='lower right')
        plt.plot(g, Min, 'b-', lw=2)
        plt.show()


def deleteModels():
    path = './'
    for foldName, subfolders, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.model'):
                os.remove(os.path.join(path, filename))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # deleteModels()
    # 初始化采样点并训练代理模型

    Optimum = []
    Resdisp = []

    for run in range(Optimization_param['runs_num']):
        Sample_Array, Sample_Optimum = SAiterate()
        Optimum.append(Sample_Optimum)
        Resdisp.append(Sample_Optimum[-1])
        print("Minimum Test Value = ", Sample_Optimum[-1])
        print(run)

    print("Mean value:", np.mean(Resdisp))
    print("Variance value:" + str(np.sqrt(np.var(Resdisp))) + "^2")

    df_col = problem_param['column_name'] + ['mean'] + ['variance']
    df_data = np.hstack((Optimum, np.zeros((len(Optimum), 2))))
    df_data[0, -2] = np.mean(Resdisp)
    df_data[0, 1] = np.sqrt(np.var(Resdisp))
    optimum_df = pd.DataFrame(data=df_data, columns=df_col)
    # optimum_df = pd.DataFrame(data=Optimum, columns=problem_param['column_name'])
    root_path = os.getcwd()
    csvname = root_path + '\\results\\runs\\' + str(problem_param['name']) + '\\'\
        + str(Optimization_param['fix_seed']) + '_' \
        + str(Optimization_param['sample_init_num']) + '+' \
        + str(Optimization_param['generations_num']) + '_results_' \
        + str(problem_param['name']) + '_' \
        + str(problem_param['dimension']) + 'd' + '.csv'
    if not os.path.exists(root_path + '\\results\\runs\\' + str(problem_param['name'])):
        os.makedirs(root_path + '\\results\\runs\\' + str(problem_param['name']))
    optimum_df.to_csv(csvname, encoding='gbk')
    # deleteModels()
