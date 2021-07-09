# -*- coding:utf-8 -*-
# Author： DYF

# 根据欧式距离来划分任务组
import numpy as np


class TaskGroup:

    def __init__(self, M, csv_name):
        self.M = M  # 任务个数
        self.data = np.loadtxt(csv_name) ##############'wv_test_M5.csv'
        print(self.data.shape[0])
        print('data:', self.data)  # 二维数组

        # 读取权重向量个数和目标个数
        self.wv_num = self.data.shape[0]
        self.obj_num = self.data.shape[1]
        print(self.wv_num, self.obj_num)
        print(self.data[0].shape[0])
        self.taskgroup = []   # 用于存储任务组



    def creat_taskgroup(self):

        def euclidean_distance(wv1, wv2):
            obj_n = wv1.shape[0]
            op1 = np.sqrt(np.sum(np.square(wv1 - wv2)))
            # op2 = np.linalg.norm(wv1 - wv2)
            return op1

        # 找出前m个小欧式距离对应的索引值
        def find_k_min(dis, top_k):
            arr = np.array([1, 3, 2, 4, 5])
            top_k_idx = arr.argsort()[::1][0:top_k]
            return top_k_idx

        taskgroup = []
        for i in range(self.wv_num - self.obj_num + 1):
            group = []
            distance = []
            for j in range(i+1, self.obj_num):
                distance.append(euclidean_distance(self.data[i], self.data[j]))
            #group.append(self.data[i].tolist())
            top_k_idx = find_k_min(distance, self.M)
            for k in range(len(top_k_idx)):
                group.append(self.data[i + top_k_idx[k]].tolist())
            taskgroup.append(group)

        return taskgroup

if __name__ == '__main__':
    csv_name = 'wv_test_M5.csv' #'W5D_100.csv'

    T = TaskGroup(2, csv_name)
    taskgroup1 = T.creat_taskgroup()
    print('taskgroup', np.array(taskgroup1))
    print('taskgroup shape', np.array(taskgroup1).shape)
    print('taskgroup [0]', np.array(taskgroup1)[0])
