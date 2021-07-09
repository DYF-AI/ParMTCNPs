# -*- coding:utf-8 -*-
# Author： DYF

import numpy as np


class WeightVector:
    # 对m维空间，目标方向个数H
    def __init__(self, H=5, m=3):
        self.H = H
        self.m = m
        self.stepsize = 1 / H

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_weight_vectors(self):
    #生成权均匀向量
        H = self.H
        m = self.m
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws

    def save_mv_to_file(self, mv, name='out.csv'):
    #保存为csv
        f = np.array(mv, dtype=np.float64)
        np.savetxt(fname=name, X=f)

    def save_to_csv(self):
        #测试
        m_v = self.get_weight_vectors()
        self.save_mv_to_file(m_v, 'wv_test_M5_1.csv')

#wv = WeightVector(3, 8)
#wv.save_to_csv()
#data = np.loadtxt('wv_test_M8_wv72.csv')
#print(data)
#print(data.shape)

