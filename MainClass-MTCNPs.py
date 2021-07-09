# -*- coding:utf-8 -*-
# Author： DYF
# Data: 20190608
import random
import numpy as np
import scipy.io as sv
from pyswarm import pso
from scipy.stats import norm

from TaskGroup import TaskGroup
from Tool import LHSample, normalize, get_total_point_x, get_total_point_y_3d, random_sample, insert_3d_array
from SearchSpace import SearchSpace
from WeightVector_1 import WeightVector_1
from WeightVector import WeightVector

from deap import base, creator
from deap import tools



# 主类Universe
class Universe:

    def __init__(self, func, fNoObjectives, fSearchSpaceDim, bounds, H, flag, TT=2, DV= 'PBI', max_evalution=200):   # func:函数名  flag：任务数  TT：多次运算计数
        self.TT = TT
        self.func = func
        self.fNoObjectives = fNoObjectives
        self.fSearchSpaceDim = fSearchSpaceDim
        self.bounds = bounds
        self.flag = flag  # 任务数
        self.TT = TT
        self.H = H
        self.max_evalution = max_evalution
        self.DV = DV
        self.vectorsNumber = 10

        self.vectorsNumber = 10
        self.searchSpace = SearchSpace(self.func, self.fNoObjectives, self.fSearchSpaceDim, self.bounds)  ## 实例searchSpace
        # self.weightVector = WeightVector_1(self.objectivesNumber, self.vectorsNumber)
        #self.weightVector = WeightVector(self.H, self.fNoObjectives)  # 实例权重向量的类 H控制权重向量的个数，参考毕业论文公式（3-6）

        #self.wv = self.weightVector.get_weight_vectors()   # 权重向量

        # 生成权重向量，并划分为任务组
        #self.weightVector.save_to_csv()  # 先把数据保存为csv格式
        #self.data = np.loadtxt('W5D_60.csv')  # 读取csv格式  # test。csv：均匀权重  wv_test1.csv:随机权重向量 # 需要在TaskGroup也修改
        if self.fNoObjectives == 5:
            self.data = np.loadtxt('wv_test_M5.csv')
            self.TG = TaskGroup(self.flag, 'wv_test_M5.csv')
            self.taskgroup = self.TG.creat_taskgroup()  # 三维列表
            print('taskgroup len:', len(self.taskgroup))
            print('taskgroup[0][0]:', self.taskgroup[0][0])

            self.group_len = len(self.taskgroup)
            self.group_i = 0  # 奇偶计数器数
            self.even = True  # 任务组奇数偶数
        elif self.fNoObjectives ==8:
            self.data = np.loadtxt('wv_test_M8_wv72.csv')
            self.TG = TaskGroup(self.flag, 'wv_test_M8_wv72.csv')
            self.taskgroup = self.TG.creat_taskgroup()  # 三维列表
            print('taskgroup len:', len(self.taskgroup))
            print('taskgroup[0][0]:', self.taskgroup[0][0])

            self.group_len = len(self.taskgroup)
            self.group_i = 0  # 奇偶计数器数
            self.even = True  # 任务组奇数偶数
        else:
            self.weightVector = WeightVector_1(self.fNoObjectives, self.vectorsNumber)
            self.wv = self.weightVector.init_weightVector_parego_mtcnp_wv20()

        #self.data = np.loadtxt('wv_test_M8_wv72.csv')  # 8个目标


        #self.TG = TaskGroup(2, 'W5D_60.csv')
        #self.TG = TaskGroup(2, 'wv_test_M5.csv')
        #self.TG = TaskGroup(2, 'wv_test_M8_wv72.csv')



        self.k = 9
        # 注意算法的评估次数和迭代次数是不一样的
        if self.fSearchSpaceDim <= 8:
            start_iters = 11 * self.fSearchSpaceDim - 1
        else:
            start_iters = self.k * self.fSearchSpaceDim - 1
        self.max_iters = (self.max_evalution - start_iters) / self.flag
        if self.max_iters - int(self.max_iters) != 0:
            self.max_iters = int(self.max_iters) + 1
        print('self.max_iters:', self.max_iters)
        self.max_iters = self.max_iters + start_iters

        # 一些参数
        self.iteration = 0
        self.ymin = 10000
        self.best_ever = 0
        self.ei = 10000

        # 输出数据文件
        self.output_f = open('output_MTCNP_OM_{}_f_task{}_obj{}_dim{}_{}_TT{}.txt'.format(self.func, self.flag, self.fNoObjectives, self.fSearchSpaceDim, self.DV, self.TT), 'w')
        self.output_x = open('output_MTCNP_OM_{}_x_task{}_obj{}_dim{}_{}_TT{}.txt'.format(self.func, self.flag, self.fNoObjectives, self.fSearchSpaceDim, self.DV, self.TT), 'w')

        #self.output_f = open('output_MTCNP_OM_{}_f_task{}_obj{}_dim{}_{}_TT{}.csv'.format(self.func, self.flag, self.fNoObjectives, self.fSearchSpaceDim, self.DV, self.TT), 'w')
        #self.output_x = open('output_MTCNP_OM_{}_x_task{}_obj{}_dim{}_{}_TT{}.csv'.format(self.func, self.flag, self.fNoObjectives, self.fSearchSpaceDim, self.DV, self.TT), 'w')

        if self.flag == 2:
            from multi_class_task_2 import TrainNet, PredictNet
            self.TrainNet = TrainNet
            self.PredictNet = PredictNet
        elif self.flag == 3:   # 预留三个任务的接口
            from multi_class_task_3 import TrainNet, PredictNet
            self.TrainNet = TrainNet
            self.PredictNet = PredictNet
        else:
            print('self.flag must be 2 or 3!')

    ## 这里不再直接得到聚合函数y了，而是得到两个目标函数
    def get_function_value(self,input_x): # input_x:二维列表如(87,8)
        f_value = []
        for i in range(len(input_x)):
            f_value.append(self.searchSpace.black_function(input_x[i])) ## self = [f1, f2]  append
        return f_value  ## [[f1,f2],...,[f1, f2]]

    def change_group_weight(self):
        if self.flag == 2:
            if self.even:
                newVector_one, newVector_two = self.taskgroup[self.group_i][0], self.taskgroup[self.group_i][1]
                self.group_i = self.group_i + 2
                if self.group_i >= len(self.taskgroup):
                    self.group_i = 0  # 计数器置0
                    self.even = False
            else:
                newVector_one, newVector_two = self.taskgroup[self.group_i+1][0], self.taskgroup[self.group_i+1][1]
                self.group_i = self.group_i + 2
                if self.group_i >= len(self.taskgroup)-1:
                    self.group_i = 0  # 计数器置0
                    self.even = True
            return newVector_one, newVector_two
        elif self.flag == 3:
            if self.even:
                newVector_one, newVector_two, newVector_three = self.taskgroup[self.group_i][0], self.taskgroup[self.group_i][1], self.taskgroup[self.group_i][2]
                self.group_i = self.group_i + 2
                if self.group_i >= len(self.taskgroup):
                    self.group_i = 0  # 计数器置0
                    self.even = False
            else:
                newVector_one, newVector_two, newVector_three = self.taskgroup[self.group_i][0], self.taskgroup[self.group_i][1], self.taskgroup[self.group_i][2]
                self.group_i = self.group_i + 2
                if self.group_i >= len(self.taskgroup)-1:
                    self.group_i = 0  # 计数器置0
                    self.even = True
            return newVector_one, newVector_two, newVector_three



    # 初始化，创建数据集
    def init_ParMTCNPs(self):

        debug = True
        #k = 9  # 调整初始化种群个体的数量
        # 产生11*D-1（如果是大于8维度，可修改为论文K*D-1）
        if self.fSearchSpaceDim <= 8:
            N = 11*self.fSearchSpaceDim - 1
            self.iteration = 11*self.fSearchSpaceDim - 1               # 从11*D-1开始计算评估，直到评估次数达到max_evalution

        else:
            N = self.k * self.fSearchSpaceDim - 1
            self.iteration = self.k * self.fSearchSpaceDim - 1
        D = self.fSearchSpaceDim   # 局部变量

        self.total_point_x = get_total_point_x(N, D, self.bounds)  # 产生样本 二维列表（87，8）
        print('debug', len(self.total_point_x))
        temp = np.array(self.total_point_x)
        self.total_point_x_3d = np.expand_dims(temp, axis=0)

        if debug:
            print('total_point_x_3d', self.total_point_x_3d)
            print('total_point_x_3d.shape', self.total_point_x_3d.shape)

        # 计算所有样本对应的函数值并保存到txt文件  初始化不用指定权重向量
        for i in range(len(self.total_point_x)):
            total_point_function_value = self.get_function_value(self.total_point_x)
        # 20190602 记录f1、f2出问题
        for i in range(len(self.total_point_x)):
            f_value = self.searchSpace.black_function(self.total_point_x[i])
            #self.output_f.write(str(f_value[0]) + ' ' + str(f_value[1]) + '\n')
            #self.output_x.write(str(self.total_point_x[i]) + '\n')
            if len(f_value) == 2:
                self.output_f.write(str(f_value[0]) + ' ' + str(f_value[1]) + '\n')
                self.output_x.write(str(self.total_point_x[i]) + '\n')
            elif len(f_value) == 3:
                self.output_f.write(str(f_value[0]) + ' ' + str(f_value[1]) + ' ' + str(f_value[2]) + '\n')
                self.output_x.write(str(self.total_point_x[i]) + '\n')
            elif len(f_value) == 5:
                self.output_f.write(str(f_value[0]) + ' ' + str(f_value[1]) + ' ' + str(f_value[2]) + ' '
                                    + str(f_value[3]) + ' ' + str(f_value[4]) + '\n')
                self.output_x.write(str(self.total_point_x[i]) + '\n')
            elif len(f_value) == 8:
                self.output_f.write(str(f_value[0]) + ' ' + str(f_value[1]) + ' ' + str(f_value[2]) + ' '
                                    + str(f_value[3]) + ' ' + str(f_value[4]) + ' ' + str(f_value[5]) + ' '
                                    + str(f_value[6]) + ' ' + str(f_value[7]) + '\n')
                self.output_x.write(str(self.total_point_x[i]) + '\n')


        # 归一化
        self.total_point_y_one = []
        total_point_norm_function_value, ideal_point = normalize(total_point_function_value, return_zstar=True)  # return_z:返回参考点
        #print('z', z)
        #input()
        print('no norm:', total_point_function_value)
        print('norm:',total_point_norm_function_value)
        ###################################### 可忽略 ###################################################################
        for i in range(len(total_point_norm_function_value)):
            self.total_point_y_one.append(self.searchSpace.techebycheff(total_point_norm_function_value[i]))

        self.total_point_y_two = []
        for i in range(len(self.total_point_x)):
            total_point_function_value = self.get_function_value(self.total_point_x)
            total_point_norm_function_value = normalize(total_point_function_value)
        for i in range(len(total_point_norm_function_value)):
            self.total_point_y_two.append(self.searchSpace.techebycheff(total_point_norm_function_value[i]))

        if self.flag == 3:  # 预留三个任务的接口
            self.total_point_y_three = []
            for i in range(len(self.total_point_x)):
                # self.total_point_y_one.append(self.searchSpace.black_function(self.total_point_x[i]))
                total_point_function_value = self.get_function_value(self.total_point_x)
                total_point_norm_function_value = normalize(total_point_function_value)
            for i in range(len(self.total_point_x)):
                self.total_point_y_three.append(self.searchSpace.techebycheff(total_point_norm_function_value[i]))
        ##########################################################################################################
        self.total_point_y_3d_one = get_total_point_y_3d(self.total_point_y_one)  # [[[y1],[y2]...]]
        self.total_point_y_3d_two = get_total_point_y_3d(self.total_point_y_two)  # [[[y1],[y2]...]]
        if self.flag == 3:
            self.total_point_y_3d_three = get_total_point_y_3d(self.total_point_y_three)

        if debug:
            print('total_point_y_3d_one', self.total_point_y_3d_one)
            print('total_point_y_3d_one', self.total_point_y_3d_one.shape)
            print('total_point_y_3d_two', self.total_point_y_3d_two)
            print('total_point_y_3d_two', self.total_point_y_3d_two.shape)
            if self.flag == 3:
                print('total_point_y_3d_three', self.total_point_y_3d_three)
                print('total_point_y_3d_three', self.total_point_y_3d_three.shape)

        # self.code_num = 65
        if self.fSearchSpaceDim <= 8:
            self.code_num = int((11 * self.fSearchSpaceDim - 1) * 0.75)
        else:
            self.code_num = int((self.k * self.fSearchSpaceDim - 1) * 0.75)
        self.first_train = self.TrainNet(save_dir="Model", dimension=self.fSearchSpaceDim)    # 保存训练模型
        self.train_train = self.PredictNet(load_dir="Model", dimension=self.fSearchSpaceDim)  # 加载训练模型

    def iterate_ParEGO(self):
        theta = 5  # NSGAIII
        if self.flag == 2:    # 两个任务
            # 将第一阶段的两、三个目标代码，与第二阶段五、八个代码融合(后续可以继续改进，统一)
            if self.fNoObjectives == 5 or self.fNoObjectives == 8:
                self.newVector_one, self.newVector_two = self.change_group_weight()
            else:
                self.newVector_one, self.newVector_two = self.weightVector.change_group_weight_wv20(self.iteration, self.flag)
        elif self.flag == 3:
            if self.fNoObjectives == 5 or self.fNoObjectives == 8:
                self.newVector_one, self.newVector_two, self.newVector_three = self.change_group_weight()
            else:
                self.newVector_one, self.newVector_two, self.newVector_three = self.weightVector.change_group_weight_task3_wv20(self.iteration, self.flag)

        # 生成对应权重向量的数据集（每个任务都要生成）
        # 产生任务一的数据
        self.searchSpace.fWeightVectors = self.newVector_one    ## 改变searchSpace的权重向量，以计算聚合函数值y
        print('fweightVector_one:', self.searchSpace.fWeightVectors)
        debug1 = False
        if debug1:
            print('self.total_point_x', self.total_point_x)
            print('self.total_point_x.shape', len(self.total_point_x))
        total_point_function_value_temp = self.get_function_value(self.total_point_x)

        if self.DV == 'TCH':
            total_point_norm_function_value_temp = normalize(total_point_function_value_temp)
        elif self.DV == 'PBI':
            total_point_norm_function_value_temp, ideal_point = normalize(total_point_function_value_temp, return_zstar=True)  #20200417
        self.total_point_y_one = []  ## 清空
        for i in range(len(total_point_norm_function_value_temp)):
            # techebycheff
            if self.DV == 'TCH':
                self.total_point_y_one.append(self.searchSpace.techebycheff(total_point_norm_function_value_temp[i]))
            elif self.DV == 'PBI':
                self.total_point_y_one.append(self.searchSpace.PBI(total_point_norm_function_value_temp[i], ideal_point, theta))  #20200417
        self.total_point_y_3d_one = get_total_point_y_3d(self.total_point_y_one)  # [[[y1],[y2]...]
        if debug1:
            print('total_point_y_3d_one', self.total_point_y_3d_one)
            print('total_point_y_3d_one.shape', self.total_point_y_3d_one.shape)
        del total_point_function_value_temp
        del total_point_norm_function_value_temp

        # 产生任务二的数据
        self.searchSpace.fWeightVectors = self.newVector_two  ## 改变searchSpace的权重，没问题
        print('fweightVector_two:', self.searchSpace.fWeightVectors)
        debug = False  # True
        if debug:
            print('self.total_point_x', self.total_point_x)
            print('self.total_point_x.shape', len(self.total_point_x))
        # self.total_point_y_two = self.get_function_value(self.total_point_x) ## [y1,y2..,yn]  ## 可以读取f1，f2
        total_point_function_value_temp = self.get_function_value(self.total_point_x)
        if self.DV == 'TCH':
            total_point_norm_function_value_temp = normalize(total_point_function_value_temp)
        elif self.DV == 'PBI':
            total_point_norm_function_value_temp, ideal_point = normalize(total_point_function_value_temp, return_zstar=True)  # 20200417
        self.total_point_y_two = []  ## 清空
        for i in range(len(total_point_norm_function_value_temp)):
            if self.DV == 'TCH':
                self.total_point_y_two.append(self.searchSpace.techebycheff(total_point_norm_function_value_temp[i]))
            elif self.DV == 'PBI':
                self.total_point_y_two.append(self.searchSpace.PBI(total_point_norm_function_value_temp[i], ideal_point, theta)) #  20200417
        self.total_point_y_3d_two = get_total_point_y_3d(self.total_point_y_two)  # [[[y1],[y2]...]
        if debug:
            print('total_point_y_3d_two', self.total_point_y_3d_two)
            print('total_point_y_3d_two.shape', self.total_point_y_3d_two.shape)
        del total_point_function_value_temp
        del total_point_norm_function_value_temp

        # 产生任务三的数据
        # 如果有任务三（至少两个任务）
        if self.flag == 3:
            ## 产生任务三的数据
            self.searchSpace.fWeightVectors = self.newVector_three  ## 改变searchSpace的权重，没问题
            print('fweightVector_one:', self.searchSpace.fWeightVectors)
            debug = False  # True
            ## 每次都要更新 y\
            if debug:
                print('self.total_point_x', self.total_point_x)
                print('self.total_point_x.shape', len(self.total_point_x))
            # self.total_point_y_three = self.get_function_value(self.total_point_x) ## [y1,y2..,yn]  ## 可以读取f1，f2
            total_point_function_value_temp = self.get_function_value(self.total_point_x)
            if self.DV == 'TCH':
                total_point_norm_function_value_temp = normalize(total_point_function_value_temp)
            elif self.DV == 'PBI':
                total_point_norm_function_value_temp, ideal_point = normalize(total_point_function_value_temp, return_zstar=True) # 20200417
            self.total_point_y_three = []  ## 清空
            for i in range(len(total_point_norm_function_value_temp)):
                if self.DV == 'TCH':
                    self.total_point_y_three.append(self.searchSpace.techebycheff(total_point_norm_function_value_temp[i]))
                elif self.DV == 'PBI':
                    self.total_point_y_three.append(self.searchSpace.PBI(total_point_norm_function_value_temp[i], ideal_point, theta))  # 20200417

            self.total_point_y_3d_three = get_total_point_y_3d(self.total_point_y_three)  # [[[y1],[y2]...]
            if debug:
                print('total_point_y_3d_three', self.total_point_y_3d_three)
                print('total_point_y_3d_three.shape', self.total_point_y_3d_three.shape)
            del total_point_function_value_temp
            del total_point_norm_function_value_temp

        ## building MTCNP model 先创建网络(cpde_num:编码器输入个数， decode_num:解码器输入个数)
        self.first_train.set_param(code_num=self.code_num, decode_num=self.total_point_x_3d.shape[1])  ### 根据输入点数，搭建网络
        # 训练网络
        for j in range(10):
            ##从总的点随机抽取0.75 作为参考点，并计算对应的y值，再打包在一起，产生一个适合输入的三维矩阵
            def update_observe_point(total_point_x_3d):  # 都是局部变量
                # 抽取3/4   20190416选取参考点，是排序找出前0.75个，还是随机找出0.75个
                observe_point_x_3d = random_sample(total_point_x_3d, 0.75)  # 从总的点中，选取0.75的点作为参考点
                # 降维
                temp = np.reshape(observe_point_x_3d, (-1, observe_point_x_3d.shape[2]))  ## (x, 8)
                # 变为二维列表
                temp1 = temp.tolist()  # [ []...[] ]
                # observe_point_y = self.get_function_value(temp1)  ##一维列表
                observe_point_y = []
                observe_point_function_value_temp = self.get_function_value(temp1)
                if self.DV == 'TCH':
                    observe_point_norm_function_value_temp = normalize(observe_point_function_value_temp)
                elif self.DV == 'PBI':
                    observe_point_norm_function_value_temp, ideal_point = normalize(observe_point_function_value_temp, return_zstar=True)  # 20200417
                for i in range(len(observe_point_norm_function_value_temp)):
                    if self.DV == 'TCH':
                        observe_point_y.append(self.searchSpace.techebycheff(observe_point_norm_function_value_temp[i]))
                    elif self.DV == 'PBI':
                        observe_point_y.append(self.searchSpace.PBI(observe_point_norm_function_value_temp[i], ideal_point, theta))
                ## 打包成一个点（x，y）
                observe_point = []
                for i in range(len(temp1)):
                    temp2 = temp1[i]  ## 要注意引用是否会影响原来的数值，copy（）
                    temp2.append(observe_point_y[i])
                    observe_point.append(temp2)
                observe_point_3d = np.expand_dims(observe_point, axis=0)
                return observe_point_3d

            ## 任务组一
            self.searchSpace.fWeightVectors = self.newVector_one  ## 每次都要改变任务对应的权重向量
            observe_point_3d_one = update_observe_point(self.total_point_x_3d)
            debug = False  # True
            if debug:
                print('Testing task group weightvector in update_observe_point! ')
                print('self.newVector_one:', self.searchSpace.fWeightVectors)
                print('observe_point_3d_one', observe_point_3d_one)
                print('observe_point_3d_one.shape', observe_point_3d_one.shape)

            ## 任务组二
            self.searchSpace.fWeightVectors = self.newVector_two  ## 每次都要改变任务对应的权重向量
            observe_point_3d_two = update_observe_point(self.total_point_x_3d)
            if debug:
                print('Testing task group weightvector in update_observe_point! ')
                print('self.newVector_two:', self.searchSpace.fWeightVectors)
                print('observe_point_3d_two', observe_point_3d_two)
                print('observe_point_3d_two.shape', observe_point_3d_two.shape)

            ## 任务组三
            if self.flag == 3:
                self.searchSpace.fWeightVectors = self.newVector_three  ## 每次都要改变任务对应的权重向量
                observe_point_3d_three = update_observe_point(self.total_point_x_3d)
                if debug:
                    print('Testing task group weightvector in update_observe_point! ')
                    print('self.newVector_three:', self.searchSpace.fWeightVectors)
                    print('observe_point_3d_three', observe_point_3d_three)
                    print('observe_point_3d_three.shape', observe_point_3d_three.shape)

            # print('observe_point_3d',observe_point_3d.shape)

            # 进行训练
            if self.flag == 2:
                self.first_train.cnp_train_model_1(observe_point_3d_one, self.total_point_x_3d,
                                                   self.total_point_y_3d_one,
                                                   observe_point_3d_two, self.total_point_x_3d,
                                                   self.total_point_y_3d_two)
            elif self.flag == 3:  ## pass
                self.first_train.cnp_train_model_1(observe_point_3d_one, self.total_point_x_3d,
                                                   self.total_point_y_3d_one,
                                                   observe_point_3d_two, self.total_point_x_3d,
                                                   self.total_point_y_3d_two,
                                                   observe_point_3d_three, self.total_point_x_3d,
                                                   self.total_point_y_3d_three)
            else:
                print('flag must be 2 or 3！')

        self.first_train.close_sess()

        if self.flag == 2:
            self.train_train.set_train_point_1(observe_point_3d_one.shape[1], observe_point_3d_one,
                                               self.total_point_y_3d_one,
                                               observe_point_3d_two, self.total_point_y_3d_two)
        elif self.flag == 3:
            self.train_train.set_train_point_1(observe_point_3d_one.shape[1], observe_point_3d_one,
                                               self.total_point_y_3d_one,
                                               observe_point_3d_two, self.total_point_y_3d_two, observe_point_3d_three,
                                               self.total_point_y_3d_three)
        # 每个任务都要计算EI值
        # 任务一的采样函数
        def evaluate_one(param):

            self.train_train.set_predict_point_1(param, task=1)
            ## 将param,放进预测，转成对应的三维数组结构
            ##predict_x = np.zeros([1, 1, 8])
            predict_x = np.zeros([1, 1, self.fSearchSpaceDim])
            for j, value in enumerate(param):
                predict_x[0, 0, j] = value
            # print('observe_point_3d_one.shape:',observe_point_3d_one.shape)
            # print('observe_point_3d_two.shape:',observe_point_3d_two.shape)
            if self.flag == 2:

                mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_two,
                                                                       predict_x)  ## 得到预测值mean和标准差sigma_one
                # mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_one, predict_x)  ## 20190510 预测：同观测到、输入
                # print('mean:',mean)
                # print('sigma:', sigma)
            elif self.flag == 3:
                mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_two,
                                                                       observe_point_3d_three, predict_x)
                # mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_one,observe_point_3d_one, predict_x)

            # print('ymin:',np.min(self.total_point_y_3d_one,axis=1))
            ymin = np.min(self.total_point_y_3d_one, axis=1)
            ei = 0
            sdis = 0
            sden = 0

            # 20190608 不同于GP包，GP包返回的就直接是标准差，而CNP返回的是方差啊？
            sigma_one = sigma_one ** 0.5

            if sigma_one <= 0:
                return 0
            if (ymin - mean) / sigma_one < -7.0:
                sdis = 0.0
            elif (ymin - mean) / sigma_one > 7.0:
                sdis = 1.0
            else:
                sdis = norm.cdf((ymin - mean) / sigma_one)
            sden = norm.pdf((ymin - mean) / sigma_one)
            ei = (ymin - mean) * sdis + sigma_one * sden
            return -ei

        def evaluate_two(param):
            self.train_train.set_predict_point_1(param, task=2)
            ## 将param,放进预测，转成对应的三维数组结构
            ##predict_x = np.zeros([1, 1, 8])
            predict_x = np.zeros([1, 1, self.fSearchSpaceDim])
            for j, value in enumerate(param):
                predict_x[0, 0, j] = value
            if self.flag == 2:
                mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_two,
                                                                       predict_x)  ## 得到预测值mean和标准差sigma_one
                # mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_one, predict_x)

            elif self.flag == 3:
                mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_two,
                                                                       observe_point_3d_three, predict_x)
                # mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_one,observe_point_3d_one, predict_x)

            # print('ymin:',np.min(self.total_point_y_3d_two,axis=1))
            ymin = np.min(self.total_point_y_3d_two, axis=1)

            # 20190608 不同于GP包，GP包返回的就直接是标准差，而CNP返回的是方差啊？
            sigma_one = sigma_one ** 0.5

            ei = 0
            sdis = 0
            sden = 0
            if sigma_one <= 0:
                return 0
            if (ymin - mean) / sigma_one < -7.0:
                sdis = 0.0
            elif (ymin - mean) / sigma_one > 7.0:
                sdis = 1.0
            else:
                sdis = norm.cdf((ymin - mean) / sigma_one)
            sden = norm.pdf((ymin - mean) / sigma_one)
            ei = (ymin - mean) * sdis + sigma_one * sden
            return -ei

        def evaluate_three(param):
            self.train_train.set_predict_point_1(param, task=3)
            ## 将param,放进预测，转成对应的三维数组结构
            ##predict_x = np.zeros([1, 1, 8])
            predict_x = np.zeros([1, 1, self.fSearchSpaceDim])
            for j, value in enumerate(param):
                predict_x[0, 0, j] = value
            if self.flag == 2:
                mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_two,
                                                                       predict_x)  ## 得到预测值mean和标准差sigma_one
                # mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_one, predict_x)

            elif self.flag == 3:
                mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_two,
                                                                       observe_point_3d_three, predict_x)
                # mean, sigma_one = self.train_train.cnp_predict_model_1(observe_point_3d_one, observe_point_3d_one,observe_point_3d_one, predict_x)

            # print('ymin:',np.min(self.total_point_y_3d_three,axis=1))
            ymin = np.min(self.total_point_y_3d_three, axis=1)

            # 20190608 不同于GP包，GP包返回的就直接是标准差，而CNP返回的是方差啊？
            sigma_one = sigma_one ** 0.5

            ei = 0
            sdis = 0
            sden = 0
            if sigma_one <= 0:
                return 0
            if (ymin - mean) / sigma_one < -7.0:
                sdis = 0.0
            elif (ymin - mean) / sigma_one > 7.0:
                sdis = 1.0
            else:
                sdis = norm.cdf((ymin - mean) / sigma_one)
            sden = norm.pdf((ymin - mean) / sigma_one)
            ei = (ymin - mean) * sdis + sigma_one * sden
            return -ei

        # 直接在这里使用遗传算法
        # Types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Initialization
        IND_SIZE = self.fSearchSpaceDim  # 种群数(维数)
        toolbox = base.Toolbox()
        toolbox.register("attribute", random.random)  ## 调用randon.random为每一个基因编码编码创建 随机初始值 也就是范围[0,1]
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operators
        # difine evaluate function # Note that a comma is a must(逗号是必须的)
        low = []
        up = []
        for i in range(len(self.bounds)):
            low.append(self.bounds[i][0])
        for i in range(len(self.bounds)):
            up.append(self.bounds[i][1])

        print('computer {}'.format(str(self.func)))
        # 将low和up处理后，后续可以合并代码
        for task in range(1, self.flag + 1):
            if self.func == 'ZDT1' or self.func == 'ZDT2' or self.func == 'ZDT3' or self.func == 'ZDT4' or self.func == 'ZDT6':
                #toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=[0, 0, 0, 0, 0, 0, 0, 0], up=[1, 1, 1, 1, 1, 1, 1, 1])
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=low, up=up)
                toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1,indpb=0.1)  # mutate : 变异                                                                   #tools.mutPolynomialBounded 多项式变异
                # toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=[0,0,0,0,0,0,0,0],up=[1,1,1,1,1,1,1,1], indpb=1.0/len(low))
                toolbox.register("select", tools.selTournament, tournsize=3)  # select : 选择保留的最佳个体
                # toolbox.register("evaluate", ga_target_function)  # commit our evaluate
                if task == 1:
                    toolbox.register("evaluate_one", evaluate_one)
                elif task == 2:
                    toolbox.register("evaluate_two", evaluate_two)
                elif task == 3:
                    toolbox.register("evaluate_three", evaluate_three)

            elif self.func == 'UF1' or self.func == 'UF2' or self.func == 'UF3' or self.func == 'UF4'or \
                    self.func == 'UF5' or self.func == 'UF6' or self.func == 'UF7' or self.func == 'UF8'\
                    or self.func == 'UF9' or self.func == 'UF10':
                #toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=[0, 0, 0, 0, 0, 0, 0, 0], up=[1, 1, 1, 1, 1, 1, 1, 1])
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=low, up=up)
                toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1,indpb=0.1)  # mutate : 变异                                                                   #tools.mutPolynomialBounded 多项式变异
                # toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=[0,0,0,0,0,0,0,0],up=[1,1,1,1,1,1,1,1], indpb=1.0/len(low))
                toolbox.register("select", tools.selTournament, tournsize=3)  # select : 选择保留的最佳个体
                # toolbox.register("evaluate", ga_target_function)  # commit our evaluate
                if task == 1:
                    toolbox.register("evaluate_one", evaluate_one)
                elif task == 2:
                    toolbox.register("evaluate_two", evaluate_two)
                elif task == 3:
                    toolbox.register("evaluate_three", evaluate_three)

            elif self.func == 'F1' or self.func == 'F2' or self.func == 'F3' or self.func == 'F4' or self.func == 'F5' \
                    or self.func == 'F6' or self.func == 'F7' or self.func == 'F8':
                #toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=[0, 0, 0, 0, 0, 0, 0, 0], up=[1, 1, 1, 1, 1, 1, 1, 1])
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=low, up=up)
                toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1,indpb=0.1)  # mutate : 变异                                                                   #tools.mutPolynomialBounded 多项式变异
                # toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=[0,0,0,0,0,0,0,0],up=[1,1,1,1,1,1,1,1], indpb=1.0/len(low))
                toolbox.register("select", tools.selTournament, tournsize=3)  # select : 选择保留的最佳个体
                # toolbox.register("evaluate", ga_target_function)  # commit our evaluate
                if task == 1:
                    toolbox.register("evaluate_one", evaluate_one)
                elif task == 2:
                    toolbox.register("evaluate_two", evaluate_two)
                elif task == 3:
                    toolbox.register("evaluate_three", evaluate_three)

            elif self.func == 'DTLZ1' or self.func == 'DTLZ2' or self.func == 'DTLZ3' or self.func == 'DTLZ4' \
                    or self.func == 'DTLZ5' or self.func == 'DTLZ6' or self.func == 'DTLZ7':
                #toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=[0, 0, 0, 0, 0, 0, 0, 0], up=[1, 1, 1, 1, 1, 1, 1, 1])
                toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=low, up=up)
                toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # mutate : 变异                                                                   #tools.mutPolynomialBounded 多项式变异
                # toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=[0,0,0,0,0,0,0,0],up=[1,1,1,1,1,1,1,1], indpb=1.0/len(low))
                toolbox.register("select", tools.selTournament, tournsize=3)  # select : 选择保留的最佳个体
                # toolbox.register("evaluate", ga_target_function)  # commit our evaluate
                #toolbox.register("evaluate", evaluate)
                if task == 1:
                    toolbox.register("evaluate_one", evaluate_one)
                elif task == 2:
                    toolbox.register("evaluate_two", evaluate_two)
                elif task == 3:
                    toolbox.register("evaluate_three", evaluate_three)
            else:
                print('Please input correct function！')


            pop = toolbox.population(n=50)       # 可设置更合适的值，以提高速度
            CXPB, MUTPB, NGEN = 0.5, 0.2, 500    # 1000

            # Evaluate the entire population
            # fitnesses = map(toolbox.evaluate, pop)
            if task == 1:
                fitnesses = map(toolbox.evaluate_one, pop)
            elif task == 2:
                fitnesses = map(toolbox.evaluate_two, pop)
            elif task == 3:
                fitnesses = map(toolbox.evaluate_three, pop)

            # fitnesses = map(toolbox.ga_target_function, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            for g in range(NGEN):
                offspring = toolbox.select(pop, len(pop))
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # 需要防止越界
                if self.func == 'ZDT1' or self.func == 'ZDT2' or self.func == 'ZDT3' or self.func == 'ZDT6':
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[i] < 0:
                                    child1[i] = random.uniform(0, 0.3)
                                elif child1[i] > 1:
                                    child1[i] = random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child2[i] < 0:
                                    child2[i] = random.uniform(0, 0.3)
                                elif child2[i] > 1:
                                    child2[i] = random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[i] < 0:
                                    mutant[i] = random.uniform(0, 0.3)
                                elif mutant[i] > 1:
                                    mutant[i] = random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                elif self.func == 'ZDT4':
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[0] < 0:
                                    child1[0] = random.uniform(0, 0.3)
                                elif child1[0] > 1:
                                    child1[0] = random.uniform(0.5, 0.999)
                                if i >= 1:
                                    if child1[i] < -5:
                                        child1[i] = -5 + random.uniform(0, 0.3)
                                    elif child1[i] > 5:
                                        child1[i] = 5 - random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child1[0] < 0:
                                    child1[0] = random.uniform(0, 0.3)
                                elif child1[0] > 1:
                                    child1[0] = random.uniform(0.5, 0.999)
                                if i >= 1:
                                    if child1[i] < -5:
                                        child1[i] = -5 + random.uniform(0, 0.3)
                                    elif child1[i] > 5:
                                        child1[i] = 5 - random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values

                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[0] < 0:
                                    mutant[0] = random.uniform(0, 0.3)
                                elif mutant[0] > 1:
                                    mutant[0] = random.uniform(0.5, 0.999)
                                if i >= 1:
                                    if mutant[i] < -5:
                                        mutant[i] = -5 + random.uniform(0, 0.3)
                                    elif child1[i] > 5:
                                        child1[i] = 5 - random.uniform(0.5, 0.999)

                            del mutant.fitness.values

                elif self.func == 'UF1' or self.func == 'UF2' or self.func == 'UF5' or self.func == 'UF6' or self.func == 'UF7':
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[0] < 0:
                                    child1[0] = random.uniform(0, 0.3)
                                elif child1[0] > 1:
                                    child1[0] = random.uniform(0.5, 0.999)
                                elif i >= 1:
                                    if child1[i] < -1:
                                        child1[i] = -1 + random.uniform(0, 0.3)
                                    elif child1[i] > 1:
                                        child1[i] = 1 - random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child2[0] < 0:
                                    child2[0] = random.uniform(0, 0.3)
                                elif child2[0] > 1:
                                    child2[0] = random.uniform(0.5, 0.999)
                                elif i >= 1:
                                    if child2[i] < -1:
                                        child2[i] = -1 + random.uniform(0, 0.3)
                                    elif child2[i] > 1:
                                        child2[i] = 1 - random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[0] < 0:
                                    mutant[0] = random.uniform(0, 0.3)
                                elif mutant[0] > 1:
                                    mutant[0] = random.uniform(0.5, 0.999)
                                elif i >= 1:
                                    if mutant[i] < -1:
                                        mutant[i] = -1 + random.uniform(0, 0.3)
                                    elif mutant[i] > 1:
                                        mutant[i] = 1 - random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                elif self.func == 'UF3':
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[i] < 0:
                                    child1[i] = random.uniform(0, 0.3)
                                elif child1[i] > 1:
                                    child1[i] = random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child2[i] < 0:
                                    child2[i] = random.uniform(0, 0.3)
                                elif child2[i] > 1:
                                    child2[i] = random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[i] < 0:
                                    mutant[i] = random.uniform(0, 0.3)
                                elif mutant[i] > 1:
                                    mutant[i] = random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                elif self.func == 'UF4':
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[0] < 0:
                                    child1[0] = random.uniform(0, 0.3)
                                elif child1[0] > 1:
                                    child1[0] = random.uniform(0.5, 0.999)
                                elif i >= 1:
                                    if child1[i] < -2:
                                        child1[i] = -2 + random.uniform(0, 0.3)
                                    elif child1[i] > 2:
                                        child1[i] = 2 - random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child1[0] < 0:
                                    child1[0] = random.uniform(0, 0.3)
                                elif child1[0] > 1:
                                    child1[0] = random.uniform(0.5, 0.999)
                                elif i >= 1:
                                    if child2[i] < -2:
                                        child2[i] = -2 + random.uniform(0, 0.3)
                                    elif child2[i] > 2:
                                        child2[i] = 2 - random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[0] < 0:
                                    mutant[0] = random.uniform(0, 0.3)
                                elif mutant[0] > 1:
                                    mutant[i] = random.uniform(0.5, 0.999)
                                if i >= 1:
                                    if mutant[i] < -2:
                                        mutant[i] = -2 + random.uniform(0, 0.3)
                                    elif mutant[i] > 2:
                                        mutant[i] = 2 - random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                elif self.func == 'UF8' or self.func == 'UF9' or self.func == 'UF10':
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if i == 0 or i == 1:
                                    if child1[0] < 0:
                                        child1[0] = random.uniform(0, 0.3)
                                    elif child1[0] > 1:
                                        child1[0] = random.uniform(0.5, 0.999)
                                    elif child1[1] < 0:
                                        child1[1] = random.uniform(0, 0.3)
                                    elif child1[1] > 1:
                                        child1[1] = random.uniform(0.5, 0.999)
                                elif i >= 2:
                                    if child1[i] < -2:
                                        child1[i] = -2 + random.uniform(0, 0.3)
                                    elif child1[i] > 2:
                                        child1[i] = 2 - random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if i == 0 or i == 1:
                                    if child2[0] < 0:
                                        child2[0] = random.uniform(0, 0.3)
                                    elif child2[0] > 1:
                                        child2[0] = random.uniform(0.5, 0.999)
                                    elif child2[1] < 0:
                                        child2[1] = random.uniform(0, 0.3)
                                    elif child2[1] > 1:
                                        child2[1] = random.uniform(0.5, 0.999)
                                elif i >= 2:
                                    if child2[i] < -2:
                                        child2[i] = -2 + random.uniform(0, 0.3)
                                    elif child2[i] > 2:
                                        child2[i] = 2 - random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if i == 0 or i == 1:
                                    if mutant[0] < 0:
                                        mutant[0] = random.uniform(0, 0.3)
                                    elif mutant[0] > 1:
                                        mutant[0] = random.uniform(0.5, 0.999)
                                    elif mutant[1] < 0:
                                        mutant[1] = random.uniform(0, 0.3)
                                    elif mutant[1] > 1:
                                        mutant[1] = random.uniform(0.5, 0.999)
                                elif i >= 2:
                                    if mutant[i] < -2:
                                        mutant[i] = -2 + random.uniform(0, 0.3)
                                    elif mutant[i] > 2:
                                        mutant[i] = 2 - random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                elif self.func == 'F1' or self.func == 'F2' or self.func == 'F3' or self.func == 'F4' or \
                        self.func == 'F5' or self.func == 'F6' or self.func == 'F7' or self.func == 'F8':
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[i] < 0:
                                    child1[i] = random.uniform(0, 0.3)
                                elif child1[i] > 1:
                                    child1[i] = random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child2[i] < 0:
                                    child2[i] = random.uniform(0, 0.3)
                                elif child2[i] > 1:
                                    child2[i] = random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[i] < 0:
                                    mutant[i] = random.uniform(0, 0.3)
                                elif mutant[i] > 1:
                                    mutant[i] = random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                elif self.func == 'F9' or self.func == 'F10':
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[0] < 0:
                                    child1[0] = random.uniform(0, 0.3)
                                elif child1[0] > 1:
                                    child1[0] = random.uniform(0.5, 0.999)
                                elif i >= 1:
                                    if child1[i] < 0:
                                        child1[i] = random.uniform(0, 0.3)
                                    elif child1[i] > 10:
                                        child1[i] = 10 - random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child1[0] < 0:
                                    child1[0] = random.uniform(0, 0.3)
                                elif child1[0] > 1:
                                    child1[0] = random.uniform(0.5, 0.999)
                                elif i >= 1:
                                    if child2[i] < 0:
                                        child2[i] = random.uniform(0, 0.3)
                                    elif child2[i] > 10:
                                        child2[i] = 10 - random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[0] < 0:
                                    mutant[0] = random.uniform(0, 0.3)
                                elif mutant[0] > 1:
                                    mutant[i] = random.uniform(0.5, 0.999)
                                if i >= 1:
                                    if mutant[i] < 0:
                                        mutant[i] = random.uniform(0, 0.3)
                                    elif mutant[i] > 10:
                                        mutant[i] = 10 - random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                elif self.func == 'DTLZ1' or self.func == 'DTLZ2' or self.func == 'DTLZ3' or \
                        self.func == 'DTLZ4' or self.func == 'DTLZ5' or self.func == 'DTLZ6' or self.func == 'DTLZ7':
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < CXPB:
                            toolbox.mate(child1, child2)
                            for i in range(len(child1)):
                                if child1[i] < 0:
                                    child1[i] = random.uniform(0, 0.3)
                                elif child1[i] > 1:
                                    child1[i] = random.uniform(0.5, 0.999)
                            for i in range(len(child2)):
                                if child2[i] < 0:
                                    child2[i] = random.uniform(0, 0.3)
                                elif child2[i] > 1:
                                    child2[i] = random.uniform(0.5, 0.999)
                            # print('child2',child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < MUTPB:
                            toolbox.mutate(mutant)
                            for i in range(len(mutant)):
                                if mutant[i] < 0:
                                    mutant[i] = random.uniform(0, 0.3)
                                elif mutant[i] > 1:
                                    mutant[i] = random.uniform(0.5, 0.999)
                            del mutant.fitness.values

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                # fitnesses = map(toolbox.evaluate, invalid_ind)
                if task == 1:
                    fitnesses = map(toolbox.evaluate_one, invalid_ind)
                elif task == 2:
                    fitnesses = map(toolbox.evaluate_two, invalid_ind)
                elif task == 3:
                    fitnesses = map(toolbox.evaluate_three, invalid_ind)

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                # The population is entirely replaced by the offspring
                pop[:] = offspring

            print("-- End of (successful) evolution --")
            best_ind = tools.selBest(pop, 1)[0]  ###  这个就是下一个需要探索的x  ,best_ind是列表
            print('best_ind:', best_ind)
            # target = self.searchSpace.black_function(best_ind)

            if task == 1:
                print('task1...')
                print('ymin:', np.min(self.total_point_y_3d_one, axis=1))
                self.searchSpace.fWeightVectors = self.newVector_one
                target_function_value = self.searchSpace.black_function(best_ind)
                target = self.searchSpace.techebycheff(target_function_value)
                # target_function_value = self.searchSpace.black_function(best_ind)
                # target_norm_function_value = normalize(target_function_value)
                # target = self.searchSpace.techebycheff(target_norm_function_value)

                # self.total_point_x_3d_one = insert_3d_array(self.total_point_x_3d, np.array(best_ind))  ## 将解添加到解码器
                self.total_point_x_3d = insert_3d_array(self.total_point_x_3d, np.array(best_ind))  ## 将解添加到解码器
                self.total_point_x.append(best_ind)
                ##self.total_point_y_3d_one = insert_3d_array(self.total_point_y_3d_one, np.array(self.searchSpace.black_function(best_ind)))## 这里可以写f1、f2
                # 其实y也是要更新的  self.total_point_y_3在这里其实没什么必要，确实是没必要，因为y会随着权重改变二改变

                # self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + '\n')
                # self.output_x.write(str(best_ind)+ '\n')
                if self.fNoObjectives == 2:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 3:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' + str(
                        self.searchSpace.f[2]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 5:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' +
                                        str(self.searchSpace.f[2]) + ' '+ str(self.searchSpace.f[3]) + ' ' +
                                        str(self.searchSpace.f[4]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 8:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' + str(self.searchSpace.f[2]) + ' '
                                        + str(self.searchSpace.f[3]) + ' ' + str(self.searchSpace.f[4]) + ' ' + str(self.searchSpace.f[5]) + ' '
                                        + str(self.searchSpace.f[6]) +' ' + str(self.searchSpace.f[7]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')

                else:
                    print('Error!')

                print('f_task{}:'.format(task), self.searchSpace.f)
                print("Found the target value to be:", target)
                print("ei_1_task{}:".format(task), best_ind.fitness.values)

            elif task == 2:
                print('task2...')
                print('ymin:', np.min(self.total_point_y_3d_two, axis=1))
                self.searchSpace.fWeightVectors = self.newVector_two
                target_function_value = self.searchSpace.black_function(best_ind)
                target = self.searchSpace.techebycheff(target_function_value)
                # target_function_value = self.searchSpace.black_function(best_ind)
                # target_norm_function_value = normalize(target_function_value)
                # target = self.searchSpace.techebycheff(target_norm_function_value)
                # self.total_point_x_3d_two = insert_3d_array(self.total_point_x_3d, np.array(best_ind))  ## 将解添加到解码器
                self.total_point_x_3d = insert_3d_array(self.total_point_x_3d, np.array(best_ind))  ## 将解添加到解码器
                self.total_point_x.append(best_ind)
                ##self.total_point_y_3d_two = insert_3d_array(self.total_point_y_3d_two, np.array(self.searchSpace.black_function(best_ind)))## 这里可以写f1、f2
                # 其实y也是要更新的  self.total_point_y_3在这里其实没什么必要，确实是没必要，因为y会随着权重改变二改变
                # self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + '\n')
                # self.output_x.write(str(best_ind)+ '\n')
                if self.fNoObjectives == 2:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 3:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' + str(
                        self.searchSpace.f[2]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 5:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' +
                                        str(self.searchSpace.f[2]) + ' '+ str(self.searchSpace.f[3]) + ' ' +
                                        str(self.searchSpace.f[4]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 8:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' + str(self.searchSpace.f[2]) + ' '
                                        + str(self.searchSpace.f[3]) + ' ' + str(self.searchSpace.f[4]) + ' ' + str(self.searchSpace.f[5]) + ' '
                                        + str(self.searchSpace.f[6]) + ' ' + str(self.searchSpace.f[7]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                else:
                    print('Error!')
                print('f_task{}:'.format(task), self.searchSpace.f)
                print("Found the target value to be:", target)
                print("ei_1_task{}:".format(task), best_ind.fitness.values)

            elif task == 3:
                print('task3...')
                print('ymin:', np.min(self.total_point_y_3d_three, axis=1))
                self.searchSpace.fWeightVectors = self.newVector_three
                # target = self.searchSpace.black_function(best_ind)
                target_function_value = self.searchSpace.black_function(best_ind)
                target = self.searchSpace.techebycheff(target_function_value)
                #target_norm_function_value = normalize(target_function_value)
                #target = self.searchSpace.techebycheff(target_norm_function_value)
                # self.total_point_x_3d_three = insert_3d_array(self.total_point_x_3d, np.array(best_ind))  ## 将解添加到解码器
                self.total_point_x_3d = insert_3d_array(self.total_point_x_3d, np.array(best_ind))  ## 将解添加到解码器
                self.total_point_x.append(best_ind)
                ##self.total_point_y_3d_three = insert_3d_array(self.total_point_y_3d_three, np.array(self.searchSpace.black_function(best_ind)))## 这里可以写f1、f2
                # 其实y也是要更新的  self.total_point_y_3在这里其实没什么必要，确实是没必要，因为y会随着权重改变二改变
                ##self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + '\n')
                ##self.output_x.write(str(best_ind)+ '\n')
                if self.fNoObjectives == 2:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 3:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' + str(
                        self.searchSpace.f[2]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 5:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' +
                                        str(self.searchSpace.f[2]) + ' '+ str(self.searchSpace.f[3]) + ' ' +
                                        str(self.searchSpace.f[4]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                elif self.fNoObjectives == 8:
                    self.output_f.write(str(self.searchSpace.f[0]) + ' ' + str(self.searchSpace.f[1]) + ' ' + str(self.searchSpace.f[2]) + ' '
                                        + str(self.searchSpace.f[3]) + ' ' + str(self.searchSpace.f[4]) + ' ' + str(self.searchSpace.f[5]) + ' '
                                        + str(self.searchSpace.f[6]) +' ' + str(self.searchSpace.f[7]) + '\n')
                    self.output_x.write(str(best_ind) + '\n')
                else:
                    print('Error!')
                print('f_task{}:'.format(task), self.searchSpace.f)
                print("Found the target value to be:", target)
                print("ei_1_task{}:".format(task), best_ind.fitness.values)

        ## 更新编码器输入
        if self.total_point_x_3d.shape[1] * 3 % 4 == 0:
            self.code_num = int(self.total_point_x_3d.shape[1] * 3 / 4)
        else:
            self.code_num = int(self.total_point_x_3d.shape[1] * 3 / 4)

        print("--------该任务第{}次训练，训练次数达到{}次".format(self.TT, self.iteration))
        print("\n")
        self.iteration += 1
        self.train_train.close_sess()

    def excute(self):
        self.init_ParMTCNPs()
        while self.iteration < self.max_iters:
            print('\n')
            print('excute iter:', self.iteration)
            self.iterate_ParEGO()
        self.output_x.close()
        self.output_f.close()

if __name__ == '__main__':

    for TT in range(1):
        #              func, fNoObjectives, fSearchSpaceDim, bounds, H, flag, TT=2, max_evalution=200
        U = Universe('F1',                                               # func
                     2,                                                  # fNoObjectives
                     8,                                                  # fSearchSpaceDim
                     [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],  # bounds
                     #[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
                     3,                                                  # H
                     2,                                                  # flag：任务数量
                     TT,                                                 # 算法重复运行次数
                     DV='TCH',                                           # PBI, TCH
                     max_evalution=200)                                  # 每次算法评价次数
        U.excute()

