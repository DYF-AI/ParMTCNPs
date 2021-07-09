# -*- coding:utf-8 -*-
# Author： DYF

import numpy as np
import random
import math
from random import shuffle


class SearchSpace:  # 输入个体x
    def __init__(self, func, fNoObjectives, fSearchSpaceDim, bounds):  # 传入函数名称就行，不要在这里传入解xpop
        self.fun = func
        self.wv = []
        self.fNoObjectives = fNoObjectives
        self.fSearchSpaceDim = fSearchSpaceDim
        self.bounds = bounds
        # if self.fun == ('ZDT1' or 'ZDT2' or 'ZDT6')    # if self.fun == 'ZDT1' or 'ZDT2' or 'ZDT6' 这样是错误滴
        if self.fun == 'ZDT1' or self.fun == 'ZDT2' or self.fun == 'ZDT6':
            fXMin = np.zeros((1, self.fSearchSpaceDim))  # 决策变量的最小值
            fXMax = np.ones((1, self.fSearchSpaceDim))  # 决策变量的最大

            print(fXMin)
            print(fXMin.shape)

            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            print('self.fIdealObjective:', self.fIdealObjective)

            # Set the weight vectors of each objective to anything,
            # so long as it sums to 1.
            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)  # 0.1
            # self.fWeightVectors = fWeightVectors

            # testing
            print(self.fWeightVectors)
            print('hello')

            ## 尚不明其作用，C++
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
        elif (self.fun == 'ZDT3'):

            fXMin = np.zeros((1, self.fSearchSpaceDim))  # 决策变量的最小值
            fXMax = np.ones((1, self.fSearchSpaceDim))
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(-1.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)

        elif (self.fun == 'ZDT4'):

            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)


        elif self.fun == 'UF1' or self.fun == 'UF2' or self.fun == 'UF5' or self.fun == 'UF6' or self.fun == 'UF7':

            #self.bounds = [[0, 1], [-1, 1], [-1, 1]]
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)

        elif self.fun == 'UF3':
            #self.fNoObjectives = 2  # 目标个数
            #self.fSearchSpaceDim = 3

            self.bounds = [[0, 1], [0, 1], [0, 1]]
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
        elif self.fun == 'UF4':
            #self.fNoObjectives = 2  # 目标个数
            #self.fSearchSpaceDim = 3
            #self.bounds = [[0, 1], [-2, 2], [-2, 2]]
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
        elif self.fun == 'UF8' or self.fun == 'UF9' or self.fun == 'UF10':  ## 三维的先不管了
            #self.fNoObjectives = 3  # 目标个数
            #self.fSearchSpaceDim = 5
            #self.bounds = [[0, 1], [0, 1], [-2, 2], [-2, 2], [-2, 2]]
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.4)  # 0.9
            self.fWeightVectors.append(0.3)
            self.fWeightVectors.append(0.3)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)

        elif self.fun == 'F1' or self.fun == 'F2' or self.fun == 'F3' or self.fun == 'F5' or self.fun == 'F6' or self.fun == 'F7':
            #self.fNoObjectives = 2  # 目标个数
            #self.fSearchSpaceDim = 8  # 3
            #self.bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)

            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)

            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
        elif self.fun == 'F4' or self.fun == 'F8':
            #self.fNoObjectives = 3  # 目标个数
            #self.fSearchSpaceDim = 8  # 3
            #self.bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.4)  # 0.9
            self.fWeightVectors.append(0.3)
            self.fWeightVectors.append(0.3)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)

        elif self.fun == 'F9' or self.fun == 'F10':
            #self.fNoObjectives = 2  # 目标个数
            #self.fSearchSpaceDim = 3
            #self.bounds = [[0, 1], [0, 10], [0, 10]]
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.9)  # 0.9
            self.fWeightVectors.append(0.1)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(300.0)
            self.fAbsMin.append(0.0)

        elif self.fun == 'DTLZ1' or self.fun == 'DTLZ2' or self.fun == 'DTLZ3' or self.fun == 'DTLZ4' or \
                self.fun == 'DTLZ5' or self.fun == 'DTLZ6' or self.fun == 'DTLZ7':
            #self.fNoObjectives = 3  # 目标个数
            #self.fSearchSpaceDim = 8  # 3
            #self.bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
            if self.fNoObjectives == 3:
                self.fIdealObjective = []
                for i in range(self.fNoObjectives):
                    self.fIdealObjective.append(0.0)
                self.fWeightVectors = []
                self.fWeightVectors.append(0.4)  # 0.9
                self.fWeightVectors.append(0.3)
                self.fWeightVectors.append(0.3)
                self.fAbsMax = []
                self.fAbsMin = []
                self.fAbsMax.append(300.0)
                self.fAbsMin.append(0.0)
                self.fAbsMax.append(300.0)
                self.fAbsMin.append(0.0)
                self.fAbsMax.append(300.0)
                self.fAbsMin.append(0.0)
            elif self.fNoObjectives == 5:
                self.fIdealObjective = []
                self.fWeightVectors = []
                self.fAbsMax = []
                self.fAbsMin = []
                for i in range(self.fNoObjectives):
                    self.fIdealObjective.append(0.0)
                    self.fWeightVectors.append(0.2)
                    self.fAbsMax.append(300.0)
                    self.fAbsMin.append(0.0)
            elif self.fNoObjectives == 8:
                self.fIdealObjective = []
                self.fWeightVectors = []
                self.fAbsMax = []
                self.fAbsMin = []
                for i in range(self.fNoObjectives):
                    self.fIdealObjective.append(0.0)
                    self.fWeightVectors.append(0.2)
                    self.fAbsMax.append(300.0)
                    self.fAbsMin.append(0.0)

        # 增加三个实际应用的问题
        elif self.fun == 'spindle':
            self.spindle_init = True
            self.fNoObjectives = 2
            self.fSearchSpaceDim = 4
            if self.fSearchSpaceDim == 4:
                self.bounds = [[150, 200], [25, 72], [1, 4], [1, 4]]
            else:
                print("spindle decision variable dimension : 4")
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.4)  # 0.9
            self.fWeightVectors.append(0.3)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(30000.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(30000.0)
            self.fAbsMin.append(0.0)

        elif self.fun == 'weld':
            self.fNoObjectives = 2
            self.fSearchSpaceDim = 4
            if self.fSearchSpaceDim == 4:
                self.bounds = [[0.125, 5], [0.1, 10], [0.125, 5], [0.1, 10]]
            else:
                print("weld decision variable dimension : 4")
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.4)  # 0.9
            self.fWeightVectors.append(0.3)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(30000.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(30000.0)
            self.fAbsMin.append(0.0)

        elif self.fun == 'f_cnc':
            self.fNoObjectives = 2
            self.fSearchSpaceDim = 4
            if self.fSearchSpaceDim == 4:
                self.bounds = [[250, 400], [0.15, 0.55], [0.5, 6], [1, 2]]
            else:
                print("cnc decision variable dimension : 4")
            self.fIdealObjective = []
            self.fIdealObjective.append(0.0)
            self.fIdealObjective.append(0.0)
            self.fWeightVectors = []
            self.fWeightVectors.append(0.4)  # 0.9
            self.fWeightVectors.append(0.3)
            self.fAbsMax = []
            self.fAbsMin = []
            self.fAbsMax.append(30000.0)
            self.fAbsMin.append(0.0)
            self.fAbsMax.append(30000.0)
            self.fAbsMin.append(0.0)

    def getSearchDemension(self):
        return self.fSearchSpaceDim

    def getBounds(self):
        return self.bounds

    def black_function(self, xpop):  # 将输入值*argv（输入任意数量的值），改为xpop（可以是列表、字典）
        debug = False
        if debug == True:
            print('debug function-------------------')
        if self.fun == 'ZDT1':
            sum1 = 0.0
            # self.f1 = float(xpop[0])
            f1 = float(xpop[0])
            for i in range(1, self.fSearchSpaceDim):  ## self.fSearchSpaceDim
                sum1 = sum1 + xpop[i]
            g = float(1 + 9.0 * (sum1 / (self.fSearchSpaceDim - 1)))
            # self.f2 = g*(1 - (self.f1/g)**(0.5))
            f2 = g * (1 - (f1 / g) ** (0.5))
            # self.f = [self.f1, self.f2]
            self.f = [f1, f2]
            # print('f:',self.f)   # 以上代码可以写成函数


        elif self.fun == 'ZDT2':
            # self.f1 = float(xpop[0])
            f1 = float(xpop[0])
            sum1 = 0
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + xpop[i]
            if debug == True:
                print('debug sum1:', sum1)
            g = float(1 + 9.0 * (sum1 / (self.fSearchSpaceDim - 1)))
            # self.f2 = g*(1 - (self.f1/g)**2)
            f2 = g * (1 - (f1 / g) ** 2)
            # self.f = [self.f1, self.f2]
            self.f = [f1, f2]

        elif self.fun == 'ZDT3':
            f1 = float(xpop[0])
            sum1 = 0
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + xpop[i]
            g = float(1 + 9.0 * (sum1 / (self.fSearchSpaceDim - 1)))
            f2 = g * (1 - math.sqrt(xpop[0] / g) - xpop[0] * math.sin(10 * math.pi * xpop[0]) / g)
            self.f = [f1, f2]

        elif self.fun == 'ZDT4':
            f1 = float(xpop[0])
            sum1 = 0.0
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i]) ** 2 - 10 * math.cos(4 * math.pi * xpop[i])
            g = float(1 + 10 * (self.fSearchSpaceDim - 1) + sum1)
            f2 = g * (1 - (f1 / g) ** (0.5))
            self.f = [f1, f2]

        elif self.fun == 'ZDT6':
            f1 = float(1 - math.exp(-4 * xpop[0]) * (math.sin(6 * math.pi * xpop[0])) ** 6)
            sum1 = 0.0
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + xpop[i]
            g = float(1 + 9 * ((sum1 / (self.fSearchSpaceDim - 1)) ** (0.25)))
            f2 = g * (1 - (f1 / g) ** 2)
            self.f = [f1, f2]

        elif self.fun == 'UF1':
            sum1 = 0
            sum2 = 0
            count1 = 0
            count2 = 0
            yj = 0
            for j in range(2, len(xpop) + 1):
                yj = xpop[j - 1] - math.sin(6.0 * math.pi * xpop[0] + j * math.pi / len(xpop))
                yj = yj * yj
                if j % 2 == 0:
                    sum2 += yj
                    count2 += 1
                else:
                    sum1 += yj
                    count1 += 1
            if len(xpop) <= 2:
                f1 = xpop[0]
            else:
                f1 = xpop[0] + 2.0 * sum1 / count1
            f2 = 1.0 - math.sqrt(xpop[0]) + 2.0 * sum2 / count2
            self.f = [f1, f2]

        elif self.fun == 'UF2':
            sum1 = 0
            sum2 = 0
            yj = 0
            count1 = 0
            count2 = 0
            for i in range(2, len(xpop) + 1):
                if i % 2 == 0:
                    yj = xpop[i - 1] - 0.3 * xpop[0] * (xpop[0] * math.cos(
                        4.0 * (6.0 * math.pi * xpop[0] + i * math.pi / len(xpop))) + 2.0) * math.cos(
                        6.0 * math.pi * xpop[0] + i * math.pi / len(xpop))
                    sum2 += yj * yj
                    count2 += 1
                else:
                    yj = xpop[i - 1] - 0.3 * xpop[0] * (xpop[0] * math.cos(
                        4.0 * (6.0 * math.pi * xpop[0] + i * math.pi / len(xpop))) + 2.0) * math.sin(
                        6.0 * math.pi * xpop[0] + i * math.pi / len(xpop))
                    sum1 += yj * yj
                    count1 += 1
            f1 = xpop[0] + 2.0 * sum1 / count1
            f2 = 1.0 - math.sqrt(xpop[0]) + 2.0 * sum2 / count2
            self.f = [f1, f2]

        elif self.fun == 'UF3':
            sum1 = 0
            sum2 = 0
            count1 = 0
            count2 = 0
            prod1 = 1.0
            prod2 = 1.0
            yj = 0
            pj = 0
            for j in range(2, len(xpop) + 1):
                yj = xpop[j - 1] - math.pow(xpop[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / (len(xpop) - 2.0)))
                pj = math.cos(20.0 * yj * math.pi / math.sqrt(j + 0.0))
                if j % 2 == 0:
                    sum2 += yj * yj
                    prod2 *= pj
                    count2 += 1
                else:
                    sum1 += yj * yj
                    prod1 *= pj
                    count1 += 1
            f1 = xpop[0] + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / count1
            f2 = 1.0 - math.sqrt(xpop[0]) + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / count2
            self.f = [f1, f2]

        elif self.fun == 'UF4':
            sum1 = 0
            sum2 = 0
            yj = 0
            hj = 0
            count1 = 0
            count2 = 0
            for j in range(2, len(xpop) + 1):
                yj = xpop[j - 1] - math.sin(6.0 * math.pi * xpop[0] + j * math.pi / len(xpop))
                hj = math.fabs(yj) / (1.0 + math.exp(2.0 * math.fabs(yj)))
                if j % 2 == 0:
                    sum2 += hj
                    count2 += 1
                else:
                    sum1 += hj
                    count1 += 1
            f1 = xpop[0] + 2.0 * sum1 / count1
            f2 = 1.0 - xpop[0] * xpop[0] + 2.0 * sum2 / count2
            self.f = [f1, f2]

        elif self.fun == 'UF5':
            sum1 = 0
            sum2 = 0
            count1 = 0
            count2 = 0
            N = 10.0
            E = 0.1
            for j in range(2, len(xpop) + 1):
                yj = xpop[j - 1] - math.sin(6.0 * math.pi * xpop[0])
                hj = 2.0 * yj * yj - math.cos(4.0 * math.pi * yj) + 1.0
                if j % 2 == 0:
                    sum2 += hj
                    count2 += 1
                else:
                    sum1 += hj
                    count1 += 1
            # hj = 0.5*(0.5/N + E)*math.fabs(math.sin(2.0*N*math.pi*xpop[0]))           ## MATLAB代码好像前面没有乘0.5(原论文也没乘以0.5，但是C++和MOEAD代码都乘以0.5)
            hj = (0.5 / N + E) * math.fabs(math.sin(2.0 * N * math.pi * xpop[0]))
            f1 = xpop[0] + hj + 2.0 * sum1 / count1
            f2 = 1.0 - xpop[0] + hj + 2.0 * sum2 / count2
            self.f = [f1, f2]

        elif self.fun == 'UF6':
            sum1 = 0
            sum2 = 0
            count1 = 0
            count2 = 0
            N = 2.0
            E = 0.1
            for j in range(2, len(xpop) + 1):
                yj = xpop[j - 1] - math.sin(6.0 * math.pi * xpop[0] + j * math.pi / len(xpop))
                if j % 2 == 0:
                    sum2 += yj * yj
                    count2 += 1
                else:
                    sum1 += yj * yj
                    count1 += 1
            # hj = 0.5*(0.5/N + E)*sin(2.0*N*math.pi*xpop[0])       ### 0.5?????
            hj = (0.5 / N + E) * math.sin(2.0 * N * math.pi * xpop[0])
            if hj < 0.0:
                hj = 0.0
            f1 = xpop[0] + hj + 2.0 * sum1 / count1
            f2 = 1.0 - xpop[0] + hj + 2.0 * sum2 / count2
            self.f = [f1, f2]

        elif self.fun == 'UF7':
            sum1 = 0
            sum2 = 0
            count1 = 0
            count2 = 0
            for j in range(2, len(xpop) + 1):
                yj = xpop[j - 1] - math.sin(6.0 * math.pi * xpop[0] + j * math.pi / len(xpop))
                if j % 2 == 0:
                    sum2 += yj * yj
                    count2 += 1
                else:
                    sum1 += yj * yj
                    count1 += 1
            yj = math.pow(xpop[0], 0.2)
            f1 = yj + 2.0 * sum1 / count1
            f2 = 1.0 - yj + 2.0 * sum2 / count2
            self.f = [f1, f2]
        elif self.fun == 'UF8':
            sum1, sum2, sum3 = 0.0, 0.0, 0.0
            count1, count2, count3 = 0.0, 0.0, 0.0
            for j in range(3, len(xpop) + 1):
                yj = xpop[j - 1] - 2.0 * xpop[1] * math.sin(2.0 * math.pi * xpop[0] + j * math.pi / len(xpop))
                if j % 3 == 1:
                    sum1 += yj * yj
                    count1 += 1
                elif j % 3 == 2:
                    sum2 += yj * yj
                    count2 += 1
                else:
                    sum3 += yj * yj
                    count3 += 1
            f1 = math.cos(0.5 * math.pi * xpop[0]) * math.cos(0.5 * math.pi * xpop[1]) + 2.0 * sum1 / count1
            f2 = math.cos(0.5 * math.pi * xpop[0]) * math.sin(0.5 * math.pi * xpop[1]) + 2.0 * sum2 / count2
            f3 = math.sin(0.5 * math.pi * xpop[0]) + 2.0 * sum3 / count3
            self.f = [f1, f2, f3]

        elif self.fun == 'UF9':
            E = 0.1
            sum1, sum2, sum3 = 0.0, 0.0, 0.0
            count1, count2, count3 = 0, 0, 0
            for j in range(3, len(xpop) + 1):
                yj = xpop[j - 1] - 2.0 * xpop[1] * math.sin(2.0 * math.pi * xpop[0] + j * math.pi / len(xpop))
                if j % 3 == 1:
                    sum1 += yj * yj
                    count1 += 1
                elif j % 3 == 2:
                    sum2 += yj * yj
                    count2 += 1
                else:
                    sum3 += yj * yj
                    count3 += 1
            yj = (0.5 + E) * (1.0 - 4.0 * (2.0 * xpop[0] - 1.0) * (2.0 * xpop[0] - 1.0))
            if yj < 0.0:
                yj = 0.0
            f1 = 0.5 * (yj + 2 * xpop[0]) * xpop[1] + 2.0 * sum1 / count1
            f2 = 0.5 * (yj - 2 * xpop[0] + 2.0) * xpop[1] + 2.0 * sum2 / count2
            f3 = 1.0 - xpop[1] + 2.0 * sum3 / count3
            self.f = [f1, f2, f3]

        elif self.fun == 'UF10':
            sum1, sum2, sum3 = 0.0, 0.0, 0.0
            count1, count2, count3 = 0, 0, 0
            for j in range(3, len(xpop) + 1):
                yj = xpop[j - 1] - 2.0 * xpop[1] * math.sin(2.0 * math.pi * xpop[0] + j * math.pi / len(xpop))
                hj = 4.0 * yj * yj - math.cos(8.0 * math.pi * yj) + 1.0
                if j % 3 == 1:
                    sum1 += hj
                    count1 += 1
                elif j % 3 == 2:
                    sum2 += hj
                    count2 += 1
                else:
                    sum3 += hj
                    count3 += 1
            f1 = math.cos(0.5 * math.pi * xpop[0]) * math.cos(0.5 * math.pi * xpop[1]) + 2.0 * sum1 / count1
            f2 = math.cos(0.5 * math.pi * xpop[0]) * math.sin(0.5 * math.pi * xpop[1]) + 2.0 * sum2 / count2
            f3 = math.sin(0.5 * math.pi * xpop[0]) + 2.0 * sum3 / count3
            self.f = [f1, f2, f3]

        elif self.fun == 'F1':
            sum1 = 0.0
            f1 = float(xpop[0])
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - xpop[0]) ** 2
            g = float(1 + 9.0 * (sum1) / (self.fSearchSpaceDim - 1))
            f2 = g * (1 - (f1 / g) ** 0.5)
            self.f = [f1, f2]
        elif self.fun == 'F2':
            sum1 = 0.0
            f1 = float(xpop[0])
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - xpop[0]) ** 2
            g = float(1 + 9.0 * (sum1) / (self.fSearchSpaceDim - 1))
            f2 = g * (1 - (f1 / g) ** 2)
            self.f = [f1, f2]

        elif self.fun == 'F3':
            f1 = float(1 - math.exp(-4 * xpop[0]) * (math.sin(6 * math.pi * xpop[0])) ** 6)
            sum1 = 0.0
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - xpop[0]) ** 2
            g = float(1 + 9 * (sum1 / 9) ** 0.25)
            f2 = g * (1 - (f1 / g) ** 2)
            self.f = [f1, f2]
        elif self.fun == 'F4':
            sum1 = 0
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - xpop[0]) ** 2
            g = sum1
            f1 = math.cos((math.pi / 2) * xpop[0]) * math.cos((math.pi / 2) * xpop[1]) * (1 + g)
            f2 = math.cos((math.pi / 2) * xpop[0]) * math.sin((math.pi / 2) * xpop[1]) * (1 + g)
            f3 = math.sin((math.pi / 2) * xpop[0]) * (1 + g)
            self.f = [f1, f2, f3]

        elif self.fun == 'F5':
            sum1 = 0.0
            f1 = float(xpop[0])
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + ((xpop[i]) ** 2 - xpop[0]) ** 2
            g = 1 + 9 * (sum1 / (self.fSearchSpaceDim - 1))
            f2 = g * (1 - (xpop[0] / g)) ** 0.5
            self.f = [f1, f2]

        elif self.fun == 'F6':
            f1 = float((xpop[0]) ** 0.5)
            sum1 = 0.0
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + ((xpop[i]) ** 2 - xpop[0]) ** 2
            g = 1 + 9 * (sum1 / (self.fSearchSpaceDim - 1))
            f2 = g * (1 - (f1 / g) ** 2)
            self.f = [f1, f2]

        elif self.fun == 'F7':
            f1 = float(1 - math.exp(-4 * xpop[0]) * (math.sin(6 * math.pi * xpop[0])) ** 6)
            sum1 = 0.0
            for i in range(1, self.fSearchSpaceDim):
                sum1 = sum1 + ((xpop[i]) ** 2 - xpop[0]) ** 2
            g = float(1 + 9 * (sum1 / 9) ** 0.25)
            f2 = g * (1 - (f1 / g) ** 2)
            self.f = [f1, f2]
        elif self.fun == 'F8':
            sum1 = 0
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + ((xpop[i]) ** 2 - xpop[0]) ** 2
            g = sum1
            f1 = math.cos((math.pi / 2) * xpop[0]) * math.cos((math.pi / 2) * xpop[1]) * (1 + g)
            f2 = math.cos((math.pi / 2) * xpop[0]) * math.sin((math.pi / 2) * xpop[1]) * (1 + g)
            f3 = math.sin((math.pi / 2) * xpop[0]) * (1 + g)
            self.f = [f1, f2, f3]

        elif self.fun == 'F9':
            f1 = float(xpop[0])
            sum1 = 0
            mul = 1
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + ((xpop[i]) ** 2 - xpop[0]) ** 2
            for i in range(2, self.fSearchSpaceDim):
                mul = mul * math.cos((xpop[i] ** 2 - xpop[0]) / (i - 1) ** 0.5)
            g = (1 / 4000) * sum1 - mul + 2
            f2 = g(1 - (f1 / g) ** 0.5)
            self.f = [f1, f2]

        elif self.fun == 'F10':
            f1 = float(xpop[0])
            sum1 = 0
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + (((xpop[i]) ** 2 - xpop[0]) ** 2 - 10 * math.cos(2 * math.pi * (xpop[i] ** 2 - xpop[0])))
            g = 1 + 10 * (self.fSearchSpaceDim - 1) + sum1
            f2 = g * (1 - (f1 / g) ** 0.5)
            self.f = [f1, f2]

        elif self.fun == 'DTLZ1_T':
            sum1 = 0
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - 0.5) ** 2 - math.cos(20 * math.pi * (xpop[i] - 0.5))
            g = 100 * (self.fSearchSpaceDim - self.fNoObjectives + 1 + sum1)
            f1 = 0.5 * xpop[0] * xpop[1] * (1 + g)
            f2 = 0.5 * xpop[0] * (1 - xpop[1]) * (1 + g)
            f3 = 0.5 * (1 - xpop[0]) * (1 + g)
            self.f = [f1, f2, f3]

        elif self.fun == 'DTLZ1':  # form MOEA-Benchmark
            k = self.fSearchSpaceDim - self.fNoObjectives + 1
            g = 0.0
            for i in range(self.fSearchSpaceDim - k, self.fSearchSpaceDim):
                g = g + (xpop[i] - 0.5) ** 2 - math.cos(20 * math.pi * (xpop[i] - 0.5))
            g = 100 * (k + g)
            y_obj = []
            for i in range(self.fNoObjectives):
                y_obj.append(0.5 * (1.0 + g))
            for i in range(self.fNoObjectives):
                for j in range(self.fNoObjectives - (i + 1)):
                    y_obj[i] = y_obj[i] * xpop[j]
                if i != 0:
                    aux = self.fNoObjectives - (i + 1)
                    y_obj[i] = y_obj[i] * (1 - xpop[aux])
            if self.fNoObjectives == 3:
                self.f = [y_obj[0], y_obj[1], y_obj[2]]  # 目前还是测DTLZ  3个目标的
            elif self.fNoObjectives == 2:
                self.f = [y_obj[0], y_obj[1]]
            elif self.fNoObjectives == 5:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4]]
            elif self.fNoObjectives == 8:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4], y_obj[5], y_obj[6], y_obj[7]]
            else:
                print('!!ERROR')

        elif self.fun == 'DTLZ2_T':
            sum1 = 0
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - 0.5) ** 2
            g = sum1
            f1 = math.cos((math.pi / 2) * xpop[0]) * math.cos((math.pi / 2) * xpop[1]) * (1 + g)
            f2 = math.cos((math.pi / 2) * xpop[0]) * math.sin((math.pi / 2) * xpop[1]) * (1 + g)
            f3 = math.sin((math.pi / 2) * xpop[0]) * (1 + g)
            self.f = [f1, f2, f3]

        elif self.fun == 'DTLZ2':
            k = self.fSearchSpaceDim - self.fNoObjectives + 1
            g = 0.0
            for i in range(self.fSearchSpaceDim - k, self.fSearchSpaceDim):
                g = g + (xpop[i] - 0.5) ** 2
            y_obj = []
            for i in range(self.fNoObjectives):
                y_obj.append(1.0 + g)
            for i in range(self.fNoObjectives):
                for j in range(self.fNoObjectives - (i + 1)):
                    y_obj[i] = y_obj[i] * math.cos(xpop[j] * 0.5 * math.pi)
                if i != 0:
                    aux = self.fNoObjectives - (i + 1)
                    y_obj[i] = y_obj[i] * math.sin(xpop[aux] * 0.5 * math.pi)
            if self.fNoObjectives == 3:
                self.f = [y_obj[0], y_obj[1], y_obj[2]]  # 目前还是测DTLZ  3个目标的
            elif self.fNoObjectives == 2:
                self.f = [y_obj[0], y_obj[1]]
            elif self.fNoObjectives == 5:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4]]
            elif self.fNoObjectives == 8:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4], y_obj[5], y_obj[6], y_obj[7]]
            else:
                print('!!ERROR')


        elif self.fun == 'DTLZ3_Error':  # 这个写法还存在bug
            sum1 = 0
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - 0.5) ** 2 - math.cos(20 * math.pi * (xpop[i] - 0.5))
            g = 100 * (self.fSearchSpaceDim - self.fNoObjectives + 1 + sum1)
            f1 = (1 + g) * math.cos(xpop[0] * math.pi / 2) * math.cos(xpop[1] * math.pi / 2)
            f2 = (1 + g) * math.cos(xpop[0] * math.pi / 2) * math.sin(xpop[1] * math.pi / 2)
            f3 = (1 + g) * math.cos(xpop[0] * math.pi / 2) * math.sin(xpop[0] * math.pi / 2)
            self.f = [f1, f2, f3]

        # D:\每周总结\论文\Test_Problem\nsga3-master\src
        elif self.fun == 'DTLZ3':
            k = self.fSearchSpaceDim - self.fNoObjectives + 1
            g = 0
            for i in range(self.fSearchSpaceDim - k, self.fSearchSpaceDim):
                g = g + (xpop[i] - 0.5) ** 2 - math.cos(20 * math.pi * (xpop[i] - 0.5))
            g = 100 * (k + g)
            y_obj = []
            for i in range(self.fNoObjectives):
                y_obj.append(1.0 + g)
            for i in range(self.fNoObjectives):
                for j in range(self.fNoObjectives - (i + 1)):
                    y_obj[i] = y_obj[i] * math.cos(xpop[j] * 0.5 * math.pi)
                if (i != 0):
                    aux = self.fNoObjectives - (i + 1)
                    y_obj[i] = y_obj[i] * math.sin(xpop[aux] * 0.5 * math.pi)
            #self.f = [y_obj[0], y_obj[1], y_obj[2]]
            if self.fNoObjectives == 3:
                self.f = [y_obj[0], y_obj[1], y_obj[2]]  # 目前还是测DTLZ  3个目标的
            elif self.fNoObjectives == 2:
                self.f = [y_obj[0], y_obj[1]]
            elif self.fNoObjectives == 5:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4]]
            elif self.fNoObjectives == 8:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4], y_obj[5], y_obj[6], y_obj[7]]

        elif self.fun == 'DTLZ4_Error':
            sum1 = 0
            alpha = 100
            for i in range(2, self.fSearchSpaceDim):
                sum1 = sum1 + (xpop[i] - 0.5) ** 2
            g = sum1
            f1 = (1 + g) * math.cos((xpop[0] ** alpha) * math.pi / 2) * math.cos((xpop[1] ** alpha) * math.pi / 2)
            f2 = (1 + g) * math.cos((xpop[0] ** alpha) * math.pi / 2) * math.sin((xpop[1] ** alpha) * math.pi / 2)
            f3 = (1 + g) * math.cos((xpop[0] ** alpha) * math.pi / 2) * math.sin((xpop[0] ** alpha) * math.pi / 2)
            self.f = [f1, f2, f3]

        elif self.fun == 'DTLZ4':
            k = self.fSearchSpaceDim - self.fNoObjectives + 1
            g = 0.0
            alpha = 100.0
            for i in range(self.fSearchSpaceDim - k, self.fSearchSpaceDim):
                g = g + (xpop[i] - 0.5) ** 2
            y_obj = []
            for i in range(self.fNoObjectives):
                y_obj.append(1.0 + g)
            for i in range(self.fNoObjectives):
                for j in range(self.fNoObjectives - (i + 1)):
                    y_obj[i] = y_obj[i] * math.cos((xpop[j] ** alpha) * math.pi / 2.0)
                if (i != 0):
                    aux = self.fNoObjectives - (i + 1)
                    y_obj[i] = y_obj[i] * math.sin((xpop[aux] ** alpha) * math.pi / 2.0)
            if self.fNoObjectives == 3:
                self.f = [y_obj[0], y_obj[1], y_obj[2]]  # 目前还是测DTLZ  3个目标的
            elif self.fNoObjectives == 2:
                self.f = [y_obj[0], y_obj[1]]
            elif self.fNoObjectives == 5:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4]]
            elif self.fNoObjectives == 8:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4], y_obj[5], y_obj[6], y_obj[7]]
            else:
                print('!!ERROR')

        elif self.fun == 'DTLZ5':
            k = self.fSearchSpaceDim - self.fNoObjectives + 1
            g = 0.0
            for i in range(self.fSearchSpaceDim - k, self.fSearchSpaceDim):
                g = g + (xpop[i] - 0.5) ** 2
            t = math.pi / (4.0 * (1.0 + g))  ### MATLAB: /(2 + 2*Temp)
            # t = math.pi / (2.0 * (1.0 + g))  ### MATLAB: /(2 + 2*Temp)   #??????
            theta = []
            theta.append(xpop[0] * math.pi / 2.0)
            for i in range(1, self.fNoObjectives - 1):
                theta.append(t * (1.0 + 2.0 * g * xpop[i]))
            y_obj = []
            for i in range(self.fNoObjectives):
                y_obj.append(1.0 + g)
            for i in range(self.fNoObjectives):
                for j in range(self.fNoObjectives - (i + 1)):
                    y_obj[i] = y_obj[i] * math.cos(theta[j])
                if i != 0:
                    aux = self.fNoObjectives - (i + 1)
                    y_obj[i] = y_obj[i] * math.sin(theta[aux])
            if self.fNoObjectives == 3:
                self.f = [y_obj[0], y_obj[1], y_obj[2]]  # 目前还是测DTLZ  3个目标的
            elif self.fNoObjectives == 2:
                self.f = [y_obj[0], y_obj[1]]
            elif self.fNoObjectives == 5:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4]]
            elif self.fNoObjectives == 8:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4], y_obj[5], y_obj[6], y_obj[7]]
            else:
                print('!!ERROR')

        elif self.fun == 'DTLZ6':
            k = self.fSearchSpaceDim - self.fNoObjectives + 1
            g = 0.0
            for i in range(self.fSearchSpaceDim - k, self.fSearchSpaceDim):
                g = g + xpop[i] ** 0.1
            t = math.pi / (4.0 * (1.0 + g))  ### MATLAB: /(2 + 2*Temp)
            # t = math.pi / (2.0 * (1.0 + g))  ### MATLAB: /(2 + 2*Temp)   #??????
            theta = []
            theta.append(xpop[0] * math.pi / 2.0)
            for i in range(1, self.fNoObjectives - 1):
                theta.append(t * (1.0 + 2.0 * g * xpop[i]))
            y_obj = []
            for i in range(self.fNoObjectives):
                y_obj.append(1.0 + g)
            for i in range(self.fNoObjectives):
                for j in range(self.fNoObjectives - (i + 1)):
                    y_obj[i] = y_obj[i] * math.cos(theta[j])
                if i != 0:
                    aux = self.fNoObjectives - (i + 1)
                    y_obj[i] = y_obj[i] * math.sin(theta[aux])
            if self.fNoObjectives == 3:
                self.f = [y_obj[0], y_obj[1], y_obj[2]]  # 目前还是测DTLZ  3个目标的
            elif self.fNoObjectives == 5:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4]]
            elif self.fNoObjectives == 8:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4], y_obj[5], y_obj[6], y_obj[7]]
            else:
                print('!!ERROR')
        elif self.fun == 'DTLZ7':
            k = self.fSearchSpaceDim - self.fNoObjectives + 1
            g = 0.0
            for i in range(self.fSearchSpaceDim - k, self.fSearchSpaceDim):
                g = g + xpop[i]
            # 以PlatEMO为准
            g = 1 + (9.0 * g) / k   # DTLZ论文
            #print('g',g)
            #g = 1 + (9.0 * g) / self.fSearchSpaceDim  ## PlatEMO：1+9*mean(PopDec(:,M:end),2); 即处于决策变量长度

            y_obj = []
            for i in range(0, self.fNoObjectives - 1):
                y_obj.append(xpop[i])

            h = 0.0
            for i in range(0, self.fNoObjectives - 1):
                h = h + (y_obj[i] / (1.0+g)) * (1 + math.sin(3.0 * math.pi*y_obj[i]))
            h = self.fNoObjectives - h
            #print('h',h)
            #y_obj[self.fNoObjectives - 1] = (1 + g) * h
            y_obj.append((1 + g) * h)
            ## 用循环表示更好
            if self.fNoObjectives == 3:
                self.f = [y_obj[0], y_obj[1], y_obj[2]]  # 目前还是测DTLZ  3个目标的
            elif self.fNoObjectives == 5:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4]]
            elif self.fNoObjectives == 8:
                self.f = [y_obj[0], y_obj[1], y_obj[2], y_obj[3], y_obj[4], y_obj[5], y_obj[6], y_obj[7]]
            else:
                print('!!ERROR')

        elif self.fun == 'spindle':
            y_obj = []
            t_2 = int(xpop[2] + 0.5)
            t_3 = int(xpop[3] + 0.5)
            d_a_set = [80, 85, 90, 95]
            d_b_set = [75, 80, 85, 90]
            d_a = d_a_set[t_2 - 1]
            d_b = d_b_set[t_3 - 1]

            d_om = 25
            d_a1 = 80
            d_a2 = 95
            d_b1 = 75
            d_b2 = 90
            p_1 = 1.25
            p_2 = 1.05
            l_k = 150
            l_g = 200
            a = 80
            E = 210000
            FF = 10000
            del_a = 0.0054
            del_b = -0.0054
            dell = 0.01  #########del
            del_ra = -0.001
            del_rb = -0.001
            I_a = 0.049 * (d_a * d_a * d_a * d_a - xpop[1] * xpop[1] * xpop[1] * xpop[1])
            I_b = 0.049 * (d_b * d_b * d_b * d_b - xpop[1] * xpop[1] * xpop[1] * xpop[1])
            c_a = 35400 * math.pow(math.fabs(del_ra), (1 / 9)) * math.pow(d_a, (10 / 9))
            c_b = 35400 * math.pow(math.fabs(del_rb), (1 / 9)) * math.pow(d_b, (10 / 9))

            # 约束条件
            g1 = p_1 * xpop[1] - d_b
            g2 = p_2 * d_b - d_a
            g3 = math.fabs(del_a + (del_a - del_b) * a / xpop[0]) - dell
            mc = 0
            BIG = 1000000

            if (g1 > 0):
                mc += BIG
            if (g2 > 0):
                mc += BIG
            if (g3 > 0):
                mc += BIG
            #print('mc', mc)
            #print('BIG', BIG)
            # 初始化的数据不应该增加惩罚项
            f1 = math.pi / 4 * (a * (d_a * d_a - xpop[1] * xpop[1]) + xpop[0] * (d_b * d_b - xpop[1] * xpop[1]))  + mc
            f2 = FF * a * a * a / (3 * E * I_a) * (1 + (xpop[0] * I_a) / (a * I_b)) + (FF / c_a) * (
                    math.pow((1 + a / xpop[0]), 2) + (c_a * a * a) / (c_b * xpop[0] * xpop[0]))  + mc / BIG
            self.f = [f1, f2]

        elif self.fun == 'weld':
            h = xpop[0]
            l = xpop[1]
            t = xpop[2]
            b = xpop[3]

            delta = 2.1952 / (t * t * t * b)
            sigma = 504000 / (b * t * t)
            P_c = 64746.022 * (1 - 0.0282346 * t) * t * b * b * b
            tau_1 = 6000 / (math.sqrt(2) * h * l)
            tau_2 = (6000 * (14 + 0.5 * l) * (math.sqrt(0.25 * (l * l + (h + t) * (h + t))))) / (
                    2 * (0.707 * h * l * (l * l / 12 + 0.25 * (h + t) * (h + t))))
            tau = math.sqrt(tau_1 * tau_1 + tau_2 * tau_2 + (l * tau_1 * tau_2) / (math.sqrt(0.25 * (l * l + (h + t) * (h + t)))))

            g1 = tau - 13600
            g2 = sigma - 30000
            g3 = h - b
            g4 = 6000 - P_c
            mc = 0
            BIG = 50
            if (g1 > 0):
                mc += BIG
            if (g2 > 0):
                mc += BIG
            if (g3 > 0):
                mc += BIG
            if (g4 > 0):
                mc += BIG
            f1 = 1.10471 * h * h * l + 0.04811 * t * b * (14 + l) + mc
            f2 = delta + mc / BIG
            self.f = [f1, f2]

        elif self.fun == 'cnc':
            v = xpop[0]
            f1 = xpop[1]
            a = xpop[2]
            r_n = 0.8
            eta = 0.75
            T = 5.48e9 * (math.pow(f1, (-0.696)) * math.pow(v, (-3.46)) * math.pow(a, (-0.46)))
            MRR = 1000 * f1 * v * a
            FF = 6.56e3 * math.pow(f1, (0.917)) * math.pow(a, (1.10)) * math.pow(v, (-0.286))
            P = v * FF / 60000
            R = 125 * f1 * f1 / r_n

            g1 = P - eta * 10
            g2 = FF - 5000
            g3 = R - 50
            mc = 0
            BIG = 10

            if (g1 > 0):
                mc += BIG
            if (g2 > 0):
                mc += BIG
            if (g3 > 0):
                mc += BIG

            f1 = 0.2 + 219912 * (1 + (0.2 / T)) / MRR + mc / BIG
            f2 = 219912 * 100 / (MRR * T) + mc
            self.f = [f1, f2]

        else:
            print('Please input correct function!')
        return self.f

    # 输入的是一个归一化后的norm_f = [norm_f1, norm_f2]
    # 关键在于在输入前是否归一化
    # ParEGO的aug_techebycheff
    def techebycheff(self, norm_f):

        norm_techbyeff = []
        sumtef = 0
        d_max = -2147000000
        nideal = [0, 0]
        diff = 0
        for j in range(self.fNoObjectives):
            if self.fun == 'ZDT3':
                # diff = self.fWeightVectors[j]*(norm_f[j] - self.fIdealObjective[j])
                diff = self.fWeightVectors[j] * norm_f[j]
                sumtef = sumtef + diff
                if diff > d_max:
                    d_max = diff
            else:
                diff = self.fWeightVectors[j] * norm_f[j]
                sumtef = sumtef + diff
                if diff > d_max:
                    d_max = diff
        return d_max + 0.05 * sumtef

    def PBI(self, norm_f, ideal_point, theta):
        # 如果是使用归一化
        norm_f = np.array(norm_f)
        z_star = np.array(ideal_point)
        w = np.array(self.fWeightVectors)

        d1 = np.linalg.norm(np.dot((norm_f - z_star), w)) / np.linalg.norm(w)
        d2 = np.linalg.norm(norm_f- (z_star + d1 * w))

        return (d1 + theta*d2).tolist()


if __name__ == 'main':
    def PBI(fWeightVectors, norm_f, ideal_point, theta):
        # 如果是使用归一化
        norm_f = np.array(norm_f)
        z_star = np.array(ideal_point)
        w = np.array(fWeightVectors)

        d1 = np.linalg.norm(np.dot((norm_f - z_star), w)) / np.linalg.norm(w)
        d2 = np.linalg.norm(norm_f - (z_star + d1 * w))

        return (d1 + theta * d2).tolist()

    fWeightVectors = [0.1, 0.2, 0.3]
    norm_f = [0.3, 0.8, 0.9]
    ideal_point = [0.01, 0, 0]
    theta = 0.5
    pbi = PBI(fWeightVectors, norm_f, ideal_point, theta)
    print('pbi', pbi)







