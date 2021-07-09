# -*- coding:utf-8 -*-
# Author： DYF

import numpy as np

# a是三维矩阵（1，8，9）第一维必须为1，b是一维矩阵[1,2...]长度必须和a的shape{2]相等
def insert_3d_array(a,b):
    #if a.shape[0] != 1:
        #print('Invalid input！')
    #if b.shape[0] != a.shape[2]:
    #    print('b can not be inserted to a!')
    # 将三维矩阵转为二维
    a = np.reshape(a,(-1, a.shape[2]))
    a1 = np.row_stack((a,b))
    array = np.expand_dims(a1, 0)
    return array


# --------------产生样本---------------
def LHSample(N, D, bounds):  # （N，D）
    print('LHS_bounds:', bounds)
    result = np.empty([N, D])  # 返回N个样本，每个样本的维度是D
    temp = np.empty([N])
    d = float(1.0 / (N))  # 1、将每一维分成不重叠的m个区间，d为区间间隔

    for i in range(D):
        for j in range(N):  # 2、在每一维里的每一个区间上随机抽取一个点，在这个点存在temp[]中
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)  # 将temp中的数据随机打乱（np.random.shuffle(x) 现场修改序列，改变自身内容。（类似洗牌，打乱顺序））
        for j in range(N):
            result[j, i] = temp[j]  # 3、再从每一维里随机抽取（2）中选取的点，

    # 对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    result = result.tolist()  # 20190320 将

    return result



def LHSample_1(D, bounds):  # （N，D）
    '''
    :param D:参数个数（D维向量空间）
    :param bounds:参数对应范围（list）（每一维对应的取值范围）
    :param N:拉丁超立方层数 （抽样样本点数N）
    :return:样本数据 （返回N*D）
    '''
    ## 20190222 将N改为11N-1  --> 11D-1
    result = np.empty([11 * D - 1, D])  # 返回N个样本，每个样本的维度是D
    temp = np.empty([11 * D - 1])
    d = float(1.0 / (11 * D - 1))  # 1、将每一维分成不重叠的m个区间，d为区间间隔

    for i in range(D):
        for j in range(11 * D - 1):  # 2、在每一维里的每一个区间上随机抽取一个点，在这个点存在temp[]中
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size=1)[0]

        np.random.shuffle(temp)  # 将temp中的数据随机打乱（np.random.shuffle(x) 现场修改序列，改变自身内容。（类似洗牌，打乱顺序））

        for j in range(11 * D - 1):
            result[j, i] = temp[j]  # 3、再从每一维里随机抽取（2）中选取的点，

    # 对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    result = result.tolist()  # 20190320 将
    return result

def get_total_point_x(N, D, bounds):
    dataSet = LHSample(N, D, bounds)
    return dataSet

## 将一个n个元素的一维数组[1, 2, 3]，转为（1，n，1）的三维数组[[[1][2][3]]]
def get_total_point_y_3d(total_point_y):
    temp = []
    for i in range(len(total_point_y)):
        temp.append([total_point_y[i]])
    temp2 = np.array(temp)
    total_point_y_3d = np.expand_dims(temp2, axis=0)
    return total_point_y_3d

def random_sample(array_3d, rate=0.75):
    # 降维
    temp1 = np.reshape(array_3d, (-1, array_3d.shape[2]))
    temp2 = temp1.copy()
    np.random.shuffle(temp2)
    temp3 = temp2[0:int(array_3d.shape[1]*rate)]
    # 升维
    observe_point_3d = np.expand_dims(temp3, axis=0)
    return observe_point_3d

# 归一化
def normalize(data, return_zstar = False): ## 输入[[f1, f2],..., [f1, f2]]
    if len(data[1]) == 2:
        result = []
        f1 = [i[0] for i in data]
        f2 = [i[1] for i in data]
        max_f1 = max(f1) # f1，即第一列的最大值
        max_f2 = max(f2)
        min_f1 = min(f1)
        min_f2 = min(f2)

        ## 整体归一化，201906021421
        ## 每个目标做归一化是行不通的，因为这样会改变目标函数值之间的大小关系
        if max_f1 > max_f2:
            max_f2 = max_f1
        else:
            max_f1 = max_f2
        if min_f1 < min_f2:
            min_f2 = min_f1
        else:
            min_f1 = min_f2
        ## norm f1
        norm_f1 = []
        norm_f2 = []
        for i in range(len(f1)):
            norm_f1.append((f1[i] - min_f1)/(max_f1 - min_f1))
        for i in range(len(f2)):
            norm_f2.append((f2[i] - min_f2)/(max_f2 - min_f2))
        for i in range(len(norm_f1)):
            result.append([norm_f1[i], norm_f2[i]])
        if return_zstar:
            return result, [min(norm_f1), min(norm_f2)]
        else:
            return result
    elif len(data[1]) == 3:
        result = []
        f1 = [i[0] for i in data]
        f2 = [i[1] for i in data]
        f3 = [i[2] for i in data]
        max_f1 = max(f1) # f1，即第一列的最大值
        max_f2 = max(f2)
        max_f3 = max(f3)
        min_f1 = min(f1)
        min_f2 = min(f2)
        min_f3 = min(f3)
        max_f = max(max_f1, max_f2, max_f3)
        min_f = min(min_f1, min_f2, min_f3)
        ## norm f1
        norm_f1 = []
        norm_f2 = []
        norm_f3 = []
        for i in range(len(f1)):
            norm_f1.append((f1[i] - min_f)/(max_f - min_f))
        for i in range(len(f2)):
            norm_f2.append((f2[i] - min_f)/(max_f - min_f))
        for i in range(len(f3)):
            norm_f3.append((f3[i] - min_f)/(max_f - min_f))
        for i in range(len(norm_f1)):
            result.append([norm_f1[i], norm_f2[i], norm_f3[i]])
        if return_zstar:
            return result, [min(norm_f1), min(norm_f2), min(norm_f3)]
        else:
            return result

    elif len(data[1]) == 5:
        result = []
        # 抽取每个列表元素的位置
        f1 = [i[0] for i in data]
        f2 = [i[1] for i in data]
        f3 = [i[2] for i in data]
        f4 = [i[3] for i in data]
        f5 = [i[4] for i in data]
        max_f1 = max(f1) # f1，即第一列的最大值
        max_f2 = max(f2)
        max_f3 = max(f3)
        max_f4 = max(f4)
        max_f5 = max(f5)

        min_f1 = min(f1)
        min_f2 = min(f2)
        min_f3 = min(f3)
        min_f4 = min(f4)
        min_f5 = min(f5)
        max_f = max(max_f1, max_f2, max_f3, max_f4, max_f5)
        min_f = min(min_f1, min_f2, min_f3, min_f4, min_f5)

        ## norm f1
        norm_f1 = []
        norm_f2 = []
        norm_f3 = []
        norm_f4 = []
        norm_f5 = []

        for i in range(len(f1)):
            norm_f1.append((f1[i] - min_f) / (max_f - min_f))
        for i in range(len(f2)):
            norm_f2.append((f2[i] - min_f) / (max_f - min_f))
        for i in range(len(f3)):
            norm_f3.append((f3[i] - min_f) / (max_f - min_f))
        for i in range(len(f4)):
            norm_f4.append((f4[i] - min_f) / (max_f - min_f))
        for i in range(len(f5)):
            norm_f5.append((f5[i] - min_f) / (max_f - min_f))

        for i in range(len(norm_f1)):
            result.append([norm_f1[i], norm_f2[i], norm_f3[i], norm_f4[i], norm_f5[i]])
        # 返回参考点
        if return_zstar:
            return result, [min(norm_f1), min(norm_f2), min(norm_f3), min(norm_f4), min(norm_f5)]
        else:
            return result
    elif len(data[1]) == 8:
        result = []
        # 抽取每个列表元素的位置
        f1 = [i[0] for i in data]
        f2 = [i[1] for i in data]
        f3 = [i[2] for i in data]
        f4 = [i[3] for i in data]
        f5 = [i[4] for i in data]
        f6 = [i[5] for i in data]
        f7 = [i[6] for i in data]
        f8 = [i[7] for i in data]
        max_f1 = max(f1) # f1，即第一列的最大值
        max_f2 = max(f2)
        max_f3 = max(f3)
        max_f4 = max(f4)
        max_f5 = max(f5)
        max_f6 = max(f6)
        max_f7 = max(f7)
        max_f8 = max(f8)

        min_f1 = min(f1)
        min_f2 = min(f2)
        min_f3 = min(f3)
        min_f4 = min(f4)
        min_f5 = min(f5)
        min_f6 = min(f6)
        min_f7 = min(f7)
        min_f8 = min(f8)
        max_f = max(max_f1, max_f2, max_f3, max_f4, max_f5, max_f6, max_f7, max_f8)
        min_f = min(min_f1, min_f2, min_f3, min_f4, min_f5, min_f6, min_f7, min_f8)

        ## norm f1
        norm_f1 = []
        norm_f2 = []
        norm_f3 = []
        norm_f4 = []
        norm_f5 = []
        norm_f6 = []
        norm_f7 = []
        norm_f8 = []

        for i in range(len(f1)):
            norm_f1.append((f1[i] - min_f) / (max_f - min_f))
        for i in range(len(f2)):
            norm_f2.append((f2[i] - min_f) / (max_f - min_f))
        for i in range(len(f3)):
            norm_f3.append((f3[i] - min_f) / (max_f - min_f))
        for i in range(len(f4)):
            norm_f4.append((f4[i] - min_f) / (max_f - min_f))
        for i in range(len(f5)):
            norm_f5.append((f5[i] - min_f) / (max_f - min_f))
        for i in range(len(f6)):
            norm_f6.append((f6[i] - min_f) / (max_f - min_f))
        for i in range(len(f7)):
            norm_f7.append((f7[i] - min_f) / (max_f - min_f))
        for i in range(len(f8)):
            norm_f8.append((f8[i] - min_f) / (max_f - min_f))

        for i in range(len(norm_f1)):
            result.append([norm_f1[i], norm_f2[i], norm_f3[i], norm_f4[i], norm_f5[i], norm_f6[i], norm_f7[i], norm_f8[i]])
        # 返回参考点
        if return_zstar:
            return result, [min(norm_f1), min(norm_f2), min(norm_f3), min(norm_f4), min(norm_f5), min(norm_f6), min(norm_f7), min(norm_f8)]
        else:
            return result