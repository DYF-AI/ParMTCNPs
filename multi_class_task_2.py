# -*- coding:utf-8 -*-
# Author： DYF
import numpy as np
import tensorflow as tf
import os

## 20190505 开始进行ParEGO_MTCNP
## 两个任务

add_layer = False#True
using_activate_func = False #True

if using_activate_func:
    assert add_layer

add_task_layer = False#True
releate_node = 16  

class TrainNet:
    CON_LAYERS = 256  # 节点数
    TRAIN_STEP = 2000

    def __init__(self, save_dir, dimension):  ## task = 2 or 3
        self.save_dir = save_dir
        self.test_dimension = dimension
        #self.task = task
        
    def close_sess(self):
        tf.reset_default_graph()
        self.sess.close()


    def set_param(self, code_num, decode_num):
        self.pointsCodeTrainNums = code_num
        self.pointsDecodeTrainNums = decode_num

        self.param_dict = {}
        # 網絡部分
        self.sess = tf.InteractiveSession()

        # 任务一的网络
        self.x_one = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.test_dimension+1])
        self.y_one = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, 1])
        # 预测x的占位符,修改。
        self.x_t_one = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, self.test_dimension])
        # 输入格式变化
        self.input_x_one = tf.reshape(self.x_one, [-1, self.test_dimension+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        self.w_fc111 = tf.Variable(tf.truncated_normal([self.test_dimension+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc111 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc111 = tf.nn.relu(tf.matmul(self.input_x_one, self.w_fc111) + self.b_fc111)

        # 编码器第二层
        self.w_fc112 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc112 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc112 = tf.nn.relu(tf.matmul(self.h_fc111, self.w_fc112) + self.b_fc112)

        # 编码器第三层
        self.w_fc113 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc113 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc113 = tf.matmul(self.h_fc112, self.w_fc113) + self.b_fc113
        self.h_fc113 = tf.reshape(self.h_fc113, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc113 = tf.reduce_mean(self.h_fc113, axis=1)

        # 编码器输出数据处理
        self.representation_one = tf.tile(tf.expand_dims(self.h_fc113, axis=1), [1, self.pointsDecodeTrainNums, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec_one = tf.concat([self.representation_one, self.x_t_one], axis=-1)
        self.dec_one = tf.reshape(self.dec_one, [-1, TrainNet.CON_LAYERS + self.test_dimension])

        # 解码器第一层
        self.w_fc121 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.test_dimension, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc121 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc121 = tf.nn.relu(tf.matmul(self.dec_one, self.w_fc121) + self.b_fc121)

        # 解码器第二层
        self.w_fc122 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc122 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc122 = tf.nn.relu(tf.matmul(self.h_fc121, self.w_fc122) + self.b_fc122)

        # 解码器第三层
        self.w_fc123 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc123 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc123 = tf.nn.relu(tf.matmul(self.h_fc122, self.w_fc123) + self.b_fc123)

        # 解码器第四层
        self.w_fc124 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc124 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc124 = tf.nn.relu(tf.matmul(self.h_fc123, self.w_fc124) + self.b_fc124)

        # 解码器第五层方差
        self.w_fc125 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc125 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_one = tf.matmul(self.h_fc124, self.w_fc125) + self.b_fc125
        self.log_v_one = tf.reshape(self.log_v_one, [-1, self.pointsDecodeTrainNums, 1])

        # 解码器第五层均值
        self.w_fc126 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc126 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_one = tf.matmul(self.h_fc124, self.w_fc126) + self.b_fc126
        self.m_one = tf.reshape(self.m_one, [-1, self.pointsDecodeTrainNums, 1])
        
        # 任务二---------------------------------------
        self.x_two = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.test_dimension+1])
        self.y_two = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, 1])
        # 预测x的占位符,修改。
        self.x_t_two = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, self.test_dimension])
        # 输入格式变化
        self.input_x_two = tf.reshape(self.x_two, [-1, self.test_dimension+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        self.w_fc211 = tf.Variable(tf.truncated_normal([self.test_dimension+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc211 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc211 = tf.nn.relu(tf.matmul(self.input_x_two, self.w_fc211) + self.b_fc211)

        # 编码器第二层
        self.w_fc212 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc212 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc212 = tf.nn.relu(tf.matmul(self.h_fc211, self.w_fc212) + self.b_fc212)

        # 编码器第三层
        self.w_fc213 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc213 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc213 = tf.matmul(self.h_fc212, self.w_fc213) + self.b_fc213
        self.h_fc213 = tf.reshape(self.h_fc213, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc213 = tf.reduce_mean(self.h_fc213, axis=1)

        # 编码器输出数据处理
        self.representation_two = tf.tile(tf.expand_dims(self.h_fc213, axis=1), [1, self.pointsDecodeTrainNums, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec_two = tf.concat([self.representation_two, self.x_t_two], axis=-1)
        self.dec_two = tf.reshape(self.dec_two, [-1, TrainNet.CON_LAYERS + self.test_dimension])

        # 解码器第一层
        self.w_fc221 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.test_dimension, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc221 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc221 = tf.nn.relu(tf.matmul(self.dec_two, self.w_fc221) + self.b_fc221)

        # 解码器第二层
        self.w_fc222 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc222 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc222 = tf.nn.relu(tf.matmul(self.h_fc221, self.w_fc222) + self.b_fc222)

        # 解码器第三层
        self.w_fc223 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc223 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc223 = tf.nn.relu(tf.matmul(self.h_fc222, self.w_fc223) + self.b_fc223)

        # 解码器第四层
        self.w_fc224 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc224 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc224 = tf.nn.relu(tf.matmul(self.h_fc223, self.w_fc224) + self.b_fc224)

        # 解码器第五层方差
        self.w_fc225 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc225 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_two = tf.matmul(self.h_fc224, self.w_fc225) + self.b_fc225
        self.log_v_two = tf.reshape(self.log_v_two, [-1, self.pointsDecodeTrainNums, 1])

        # 解码器第五层均值
        self.w_fc226 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc226 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_two = tf.matmul(self.h_fc224, self.w_fc226) + self.b_fc226
        self.m_two = tf.reshape(self.m_two, [-1, self.pointsDecodeTrainNums, 1])
        
        self.m_multi = tf.reshape(tf.concat([self.m_one, self.m_two], axis=-1), [-1, 2])  ## 
        self.log_v_multi = tf.reshape(tf.concat([self.log_v_one, self.log_v_two], axis=-1), [-1, 2])
        if add_layer:
            # 新的相关性 
            self.w_fc11_relu = tf.Variable(tf.truncated_normal([2, releate_node], stddev=0.1))
            self.b_fc11_relu = tf.Variable(tf.constant(0.1, shape=[releate_node]))
            self.w_fc12_relu = tf.Variable(tf.truncated_normal([2, releate_node], stddev=0.1))
            self.b_fc12_relu = tf.Variable(tf.constant(0.1, shape=[releate_node]))

            self.h_fc11_relu = tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu
            self.h_fc12_relu = tf.matmul(self.log_v_multi, self.w_fc12_relu) + self.b_fc12_relu

            self.w_fc21_one = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc21_one = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc21_two = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc21_two = tf.Variable(tf.constant(0.1, shape=[1]))
            self.m_one = tf.matmul(self.h_fc11_relu, self.w_fc21_one) + self.b_fc21_one
            self.m_two = tf.matmul(self.h_fc11_relu, self.w_fc21_two) + self.b_fc21_two
            self.m_one = tf.reshape(self.m_one, [-1, self.pointsDecodeTrainNums, 1])
            self.m_two = tf.reshape(self.m_two, [-1, self.pointsDecodeTrainNums, 1])

            self.w_fc22_one = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc22_one = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc22_two = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc22_two = tf.Variable(tf.constant(0.1, shape=[1]))
            self.log_v_one = tf.matmul(self.h_fc12_relu, self.w_fc22_one) + self.b_fc22_one
            self.log_v_two = tf.matmul(self.h_fc12_relu, self.w_fc22_two) + self.b_fc22_two
            self.log_v_one = tf.reshape(self.log_v_one, [-1, self.pointsDecodeTrainNums, 1])
            self.log_v_two = tf.reshape(self.log_v_two, [-1, self.pointsDecodeTrainNums, 1])
            

            # print("using relu...")
            # # self.w_fc11_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # # self.b_fc11_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # # self.w_fc12_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # # self.b_fc12_relu = tf.Variable(tf.constant(0.1, shape=[1]))            
            # self.w_fc11_relu = tf.Variable(tf.truncated_normal([2 , 1], mean=0.5, stddev=0.5))
            # self.b_fc11_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc12_relu = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc12_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # print("self.w_fc11_relu:",self.w_fc11_relu)



            # if using_activate_func:
            #     # self.m_one_relu = tf.nn.relu(tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu)
            #     # self.m_two_relu = tf.nn.relu(tf.matmul(self.m_multi, self.w_fc12_relu) + self.b_fc12_relu)
            #     self.m_one_relu = tf.nn.sigmoid(tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu)
            #     self.m_two_relu = tf.nn.sigmoid(tf.matmul(self.m_multi, self.w_fc12_relu) + self.b_fc12_relu)
            # else:
            #     self.m_one_relu = tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu
            #     self.m_two_relu = tf.matmul(self.m_multi, self.w_fc12_relu) + self.b_fc12_relu
            # self.m_one_relu = tf.reshape(self.m_one_relu, [-1, self.pointsDecodeTrainNums, 1])
            # self.m_two_relu = tf.reshape(self.m_two_relu, [-1, self.pointsDecodeTrainNums, 1])

            # # add relu
            # # self.w_fc21_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # # self.b_fc21_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # # self.w_fc22_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # # self.b_fc22_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc21_relu = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc21_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc22_relu = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc22_relu = tf.Variable(tf.constant(0.1, shape=[1]))

            # if using_activate_func:
            #     # self.log_v_one_relu = tf.nn.relu(tf.matmul(self.log_v_multi, self.w_fc21_relu) + self.b_fc21_relu)
            #     # self.log_v_two_relu = tf.nn.relu(tf.matmul(self.log_v_multi, self.w_fc22_relu) + self.b_fc22_relu)
            #     self.log_v_one_relu = tf.nn.sigmoid(tf.matmul(self.log_v_multi, self.w_fc21_relu) + self.b_fc21_relu)
            #     self.log_v_two_relu = tf.nn.sigmoid(tf.matmul(self.log_v_multi, self.w_fc22_relu) + self.b_fc22_relu)
            # else:
            #     self.log_v_one_relu = tf.matmul(self.log_v_multi, self.w_fc21_relu) + self.b_fc21_relu
            #     self.log_v_two_relu = tf.matmul(self.log_v_multi, self.w_fc22_relu) + self.b_fc22_relu
            # self.log_v_one_relu = tf.reshape(self.log_v_one_relu, [-1, self.pointsDecodeTrainNums, 1])
            # self.log_v_two_relu = tf.reshape(self.log_v_two_relu, [-1, self.pointsDecodeTrainNums, 1])

            # # 最后一层输出
            # self.m_multi_relu = tf.reshape(tf.concat([self.m_one_relu, self.m_two_relu], axis=-1), [-1, 2])  ## 
            # self.log_v_multi_relu = tf.reshape(tf.concat([self.log_v_one_relu, self.log_v_two_relu], axis=-1), [-1, 2])

            # # self.w_fc11 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # # self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
            # # self.w_fc12 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # # self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc11 = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc12 = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))

            # self.m_one = tf.matmul(self.m_multi_relu, self.w_fc11) + self.b_fc11
            # self.m_one = tf.reshape(self.m_one, [-1, self.pointsDecodeTrainNums, 1])
            # self.m_two = tf.matmul(self.m_multi_relu, self.w_fc12) + self.b_fc12
            # self.m_two = tf.reshape(self.m_two, [-1, self.pointsDecodeTrainNums, 1])
            # print("self.m_one:", self.m_one)

            # self.w_fc21 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc22 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))

            # self.log_v_one = tf.matmul(self.log_v_multi_relu, self.w_fc21) + self.b_fc21
            # self.log_v_one = tf.reshape(self.log_v_one, [-1, self.pointsDecodeTrainNums, 1])
            # self.log_v_two = tf.matmul(self.log_v_multi_relu, self.w_fc22) + self.b_fc22
            # self.log_v_two = tf.reshape(self.log_v_two, [-1, self.pointsDecodeTrainNums, 1])
            # print("self.log_v_one:", self.log_v_one)
        else:
            # 最后一层均值输出
            self.w_fc11 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc12 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc11 = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc12 = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))

            self.m_one = tf.matmul(self.m_multi, self.w_fc11) + self.b_fc11
            self.m_one = tf.reshape(self.m_one, [-1, self.pointsDecodeTrainNums, 1])
            self.m_two = tf.matmul(self.m_multi, self.w_fc12) + self.b_fc12
            self.m_two = tf.reshape(self.m_two, [-1, self.pointsDecodeTrainNums, 1])

            # 最后一层方差输出
            self.w_fc21 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc22 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc21 = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc22 = tf.Variable(tf.truncated_normal([2, 1], mean=0.5, stddev=0.5))
            # self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))

            self.log_v_one = tf.matmul(self.log_v_multi, self.w_fc21) + self.b_fc21
            self.log_v_one = tf.reshape(self.log_v_one, [-1, self.pointsDecodeTrainNums, 1])
            self.log_v_two = tf.matmul(self.log_v_multi, self.w_fc22) + self.b_fc22
            self.log_v_two = tf.reshape(self.log_v_two, [-1, self.pointsDecodeTrainNums, 1])

        if add_task_layer:
            # ------------------------------------------------------------------
            # 在每个输出在后面添加一层
            self.w_fc11_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc11_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.w_fc12_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc12_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.m_one_ = tf.matmul(self.m_one, self.w_fc11_) + self.b_fc11_
            self.m_one_ = tf.reshape(self.m_one_, [-1, self.pointsDecodeTrainNums, 1])
            self.m_two_ = tf.matmul(self.m_two, self.w_fc12_) + self.b_fc12
            self.m_two_ = tf.reshape(self.m_two_, [-1, self.pointsDecodeTrainNums, 1])

            self.w_fc21_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc21_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.w_fc22_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc22_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.log_v_one_ = tf.matmul(self.log_v_one, self.w_fc21_) + self.b_fc21_
            self.log_v_one_ = tf.reshape(self.log_v_one_, [-1, self.pointsDecodeTrainNums, 1])
            self.log_v_two_ = tf.matmul(self.log_v_two, self.w_fc22_) + self.b_fc22_
            self.log_v_two_ = tf.reshape(self.log_v_two_, [-1, self.pointsDecodeTrainNums, 1])
            # -------------------------------------------------------------

        # 损失函数
        if add_task_layer:
            self.sigma_one = 0.1 + 0.9 * tf.nn.softplus(self.log_v_one_)
            self.dist_one = tf.contrib.distributions.MultivariateNormalDiag(loc=self.m_one_, scale_diag=self.sigma_one)
            self.log_p_one = self.dist_one.log_prob(self.y_one)
            self.loss_one = -tf.reduce_mean(self.log_p_one)
            
            self.sigma_two = 0.1 + 0.9 * tf.nn.softplus(self.log_v_two_)
            self.dist_two = tf.contrib.distributions.MultivariateNormalDiag(loc=self.m_two_, scale_diag=self.sigma_two)
            self.log_p_two = self.dist_two.log_prob(self.y_two)
            self.loss_two = -tf.reduce_mean(self.log_p_two)
        else:
            self.sigma_one = 0.1 + 0.9 * tf.nn.softplus(self.log_v_one)
            self.dist_one = tf.contrib.distributions.MultivariateNormalDiag(loc=self.m_one, scale_diag=self.sigma_one)
            self.log_p_one = self.dist_one.log_prob(self.y_one)
            self.loss_one = -tf.reduce_mean(self.log_p_one)
            
            self.sigma_two = 0.1 + 0.9 * tf.nn.softplus(self.log_v_two)
            self.dist_two = tf.contrib.distributions.MultivariateNormalDiag(loc=self.m_two, scale_diag=self.sigma_two)
            self.log_p_two = self.dist_two.log_prob(self.y_two)
            self.loss_two = -tf.reduce_mean(self.log_p_two)


        self.loss = self.loss_one + self.loss_two 

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_step = self.optimizer.minimize(self.loss)
        tf.global_variables_initializer().run()
        
     
    def cnp_train_model_1(self, observe_point_one, total_point_x_one, total_point_y_one, observe_point_two, total_point_x_two, total_point_y_two):
        # 任务一参数
        self.param_dict['fc_111w'] = self.w_fc111
        self.param_dict['fc_111b'] = self.b_fc111
        self.param_dict['fc_112w'] = self.w_fc112
        self.param_dict['fc_112b'] = self.b_fc112
        self.param_dict['fc_113w'] = self.w_fc113
        self.param_dict['fc_113b'] = self.b_fc113
        self.param_dict['fc_121w'] = self.w_fc121
        self.param_dict['fc_121b'] = self.b_fc121
        self.param_dict['fc_122w'] = self.w_fc122
        self.param_dict['fc_122b'] = self.b_fc122
        self.param_dict['fc_123w'] = self.w_fc123
        self.param_dict['fc_123b'] = self.b_fc123
        self.param_dict['fc_124w'] = self.w_fc124
        self.param_dict['fc_124b'] = self.b_fc124
        self.param_dict['fc_125w'] = self.w_fc125
        self.param_dict['fc_125b'] = self.b_fc125
        self.param_dict['fc_126w'] = self.w_fc126
        self.param_dict['fc_126b'] = self.b_fc126
        
         # 任务二参数
        self.param_dict['fc_211w'] = self.w_fc211
        self.param_dict['fc_211b'] = self.b_fc211
        self.param_dict['fc_212w'] = self.w_fc212
        self.param_dict['fc_212b'] = self.b_fc212
        self.param_dict['fc_213w'] = self.w_fc213
        self.param_dict['fc_213b'] = self.b_fc213
        self.param_dict['fc_221w'] = self.w_fc221
        self.param_dict['fc_221b'] = self.b_fc221
        self.param_dict['fc_222w'] = self.w_fc222
        self.param_dict['fc_222b'] = self.b_fc222
        self.param_dict['fc_223w'] = self.w_fc223
        self.param_dict['fc_223b'] = self.b_fc223
        self.param_dict['fc_224w'] = self.w_fc224
        self.param_dict['fc_224b'] = self.b_fc224
        self.param_dict['fc_225w'] = self.w_fc225
        self.param_dict['fc_225b'] = self.b_fc225
        self.param_dict['fc_226w'] = self.w_fc226
        self.param_dict['fc_226b'] = self.b_fc226
        
        # add relu
        if add_layer:
            self.param_dict['fc_11w_relu'] = self.w_fc11_relu
            self.param_dict['fc_11b_relu'] = self.b_fc11_relu
            self.param_dict['fc_12w_relu'] = self.w_fc12_relu
            self.param_dict['fc_12b_relu'] = self.b_fc12_relu

            self.param_dict['fc_21w_one'] = self.w_fc21_one
            self.param_dict['fc_21b_one'] = self.b_fc21_one
            self.param_dict['fc_21w_two'] = self.w_fc21_two
            self.param_dict['fc_21b_two'] = self.b_fc21_two

            self.param_dict['fc_22w_one'] = self.w_fc22_one
            self.param_dict['fc_22b_one'] = self.b_fc22_one
            self.param_dict['fc_22w_two'] = self.w_fc22_two
            self.param_dict['fc_22b_two'] = self.b_fc22_two

        # 最后连接层
        self.param_dict['fc_11w'] = self.w_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.w_fc12
        self.param_dict['fc_12b'] = self.b_fc12

        self.param_dict['fc_21w'] = self.w_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.w_fc22
        self.param_dict['fc_22b'] = self.b_fc22

        if add_task_layer:
            self.param_dict['fc_11w_'] = self.w_fc11_
            self.param_dict['fc_11b_'] = self.b_fc11_
            self.param_dict['fc_12w_'] = self.w_fc12_
            self.param_dict['fc_12b_'] = self.b_fc12_

            self.param_dict['fc_21w_'] = self.w_fc21_
            self.param_dict['fc_21b_'] = self.b_fc21_
            self.param_dict['fc_22w_'] = self.w_fc22_
            self.param_dict['fc_22b_'] = self.b_fc22_

        saver = tf.train.Saver(self.param_dict)
        
        # loss_value_history = []
        # 模型训练
        for i in range(TrainNet.TRAIN_STEP):
            self.train_step.run(feed_dict={self.x_one: observe_point_one, self.y_one: total_point_y_one, self.x_t_one: total_point_x_one,
                                           self.x_two: observe_point_two, self.y_two: total_point_y_two, self.x_t_two: total_point_x_two})
            ##loss_value = loss_value + self.loss.eval(feed_dict={self.x_one: observe_point_one, self.y_one: total_point_y_one, 
            ##                                                    self.x_t_one:total_point_x_one,                                                                   ##                                                      self.x_two: observe_point_two, self.y_two: total_point_y_two, 
            ##                                                    self.x_t_two: total_point_x_two}).mean()
            loss_value = self.loss.eval(feed_dict={self.x_one: observe_point_one, self.y_one: total_point_y_one, 
                                                                self.x_t_one:total_point_x_one,                                                                                                                           self.x_two: observe_point_two, self.y_two: total_point_y_two, 
                                                                self.x_t_two: total_point_x_two}).mean()
            # 5000次打印一次总loss的平均值
            if i % (TrainNet.TRAIN_STEP - 1) == 0:
                if i == 0:
                    continue
                ##print("error:第{}次 {}".format(i + 1, loss_value / (3 * TrainNet.TRAIN_STEP)))
                print("loss No {}: {}".format(i + 1, loss_value))
                # loss_value_history.append(loss_value)
                loss_value = 0
            # 如果前后loss变化少于阈值，停止训练
            # if abs(loss_value_history[-1] - loss_value_history[-2]) < 0.05:
            #     break

        # 储存训练的模型
        path = './cnp_model/{}/'.format(self.save_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(self.sess, './cnp_model/{}/cnp_model'.format(self.save_dir))
        # predict_sigma = sigma.eval(feed_dict={x: X, y: Y, x_t: predict_x})


class PredictNet:
    CON_LAYERS = 256

    def __init__(self, load_dir, dimension):
        self.load_dir = load_dir
        self.test_dimension = dimension
    
    def set_train_point_1(self,code_num, observe_point_one, total_point_y_one, observe_point_two, total_point_y_two):
        self.pointsCodeTrainNums = code_num
        self.observe_point_one = observe_point_one
        self.total_point_y_one = total_point_y_one
        self.observe_point_two = observe_point_two
        self.total_point_y_two = total_point_y_two

        self.pointsDecodeTrainNums = 1
        
        self.param_dict = {}
        # 任务一的网络
        self.x_one = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.test_dimension+1])
        self.y_one = tf.placeholder(tf.float32, [None, 1, 1])
        # 预测x的占位符,修改。
        self.x_t_one = tf.placeholder(tf.float32, [None, 1, self.test_dimension])
        # 输入格式变化
        self.input_x_one = tf.reshape(self.x_one, [-1, self.test_dimension+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        self.w_fc111 = tf.Variable(tf.truncated_normal([self.test_dimension+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc111 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc111 = tf.nn.relu(tf.matmul(self.input_x_one, self.w_fc111) + self.b_fc111)

        # 编码器第二层
        self.w_fc112 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc112 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc112 = tf.nn.relu(tf.matmul(self.h_fc111, self.w_fc112) + self.b_fc112)

        # 编码器第三层
        self.w_fc113 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc113 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc113 = tf.matmul(self.h_fc112, self.w_fc113) + self.b_fc113
        self.h_fc113 = tf.reshape(self.h_fc113, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc113 = tf.reduce_mean(self.h_fc113, axis=1)

        # 编码器输出数据处理
        self.representation_one = tf.tile(tf.expand_dims(self.h_fc113, axis=1), [1, 1, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec_one = tf.concat([self.representation_one, self.x_t_one], axis=-1)
        self.dec_one = tf.reshape(self.dec_one, [-1, TrainNet.CON_LAYERS + self.test_dimension])

        # 解码器第一层
        self.w_fc121 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.test_dimension, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc121 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc121 = tf.nn.relu(tf.matmul(self.dec_one, self.w_fc121) + self.b_fc121)

        # 解码器第二层
        self.w_fc122 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc122 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc122 = tf.nn.relu(tf.matmul(self.h_fc121, self.w_fc122) + self.b_fc122)

        # 解码器第三层
        self.w_fc123 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc123 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc123 = tf.nn.relu(tf.matmul(self.h_fc122, self.w_fc123) + self.b_fc123)

        # 解码器第四层
        self.w_fc124 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc124 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc124 = tf.nn.relu(tf.matmul(self.h_fc123, self.w_fc124) + self.b_fc124)

        # 解码器第五层方差
        self.w_fc125 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc125 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_one = tf.matmul(self.h_fc124, self.w_fc125) + self.b_fc125
        self.log_v_one = tf.reshape(self.log_v_one, [-1, 1, 1])

        # 解码器第五层均值
        self.w_fc126 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc126 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_one = tf.matmul(self.h_fc124, self.w_fc126) + self.b_fc126
        self.m_one = tf.reshape(self.m_one, [-1, 1, 1])

        # 任务二---------------------------------------
        self.x_two = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.test_dimension+1])
        self.y_two = tf.placeholder(tf.float32, [None, 1, 1])
        # 预测x的占位符,修改。
        self.x_t_two = tf.placeholder(tf.float32, [None, 1, self.test_dimension])
        # 输入格式变化
        self.input_x_two = tf.reshape(self.x_two, [-1, self.test_dimension+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        self.w_fc211 = tf.Variable(tf.truncated_normal([self.test_dimension+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc211 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc211 = tf.nn.relu(tf.matmul(self.input_x_two, self.w_fc211) + self.b_fc211)

        # 编码器第二层
        self.w_fc212 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc212 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc212 = tf.nn.relu(tf.matmul(self.h_fc211, self.w_fc212) + self.b_fc212)

        # 编码器第三层
        self.w_fc213 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc213 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc213 = tf.matmul(self.h_fc212, self.w_fc213) + self.b_fc213
        self.h_fc213 = tf.reshape(self.h_fc213, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc213 = tf.reduce_mean(self.h_fc213, axis=1)

        # 编码器输出数据处理
        self.representation_two = tf.tile(tf.expand_dims(self.h_fc213, axis=1), [1, 1, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec_two = tf.concat([self.representation_two, self.x_t_two], axis=-1)
        self.dec_two = tf.reshape(self.dec_two, [-1, TrainNet.CON_LAYERS + self.test_dimension])

        # 解码器第一层
        self.w_fc221 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.test_dimension, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc221 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc221 = tf.nn.relu(tf.matmul(self.dec_two, self.w_fc221) + self.b_fc221)

        # 解码器第二层
        self.w_fc222 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc222 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc222 = tf.nn.relu(tf.matmul(self.h_fc221, self.w_fc222) + self.b_fc222)

        # 解码器第三层
        self.w_fc223 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc223 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc223 = tf.nn.relu(tf.matmul(self.h_fc222, self.w_fc223) + self.b_fc223)

        # 解码器第四层
        self.w_fc224 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc224 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc224 = tf.nn.relu(tf.matmul(self.h_fc223, self.w_fc224) + self.b_fc224)

        # 解码器第五层方差
        self.w_fc225 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc225 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_two = tf.matmul(self.h_fc224, self.w_fc225) + self.b_fc225
        self.log_v_two = tf.reshape(self.log_v_two, [-1, 1, 1])

        # 解码器第五层均值
        self.w_fc226 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc226 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_two = tf.matmul(self.h_fc224, self.w_fc226) + self.b_fc226
        self.m_two = tf.reshape(self.m_two, [-1, 1, 1])
        
        self.m_multi = tf.reshape(tf.concat([self.m_one, self.m_two], axis=-1), [-1, 2])  ## 
        self.log_v_multi = tf.reshape(tf.concat([self.log_v_one, self.log_v_two], axis=-1), [-1, 2])
        if add_layer:
            # 新的相关性 
            self.w_fc11_relu = tf.Variable(tf.truncated_normal([2, releate_node], stddev=0.1))
            self.b_fc11_relu = tf.Variable(tf.constant(0.1, shape=[releate_node]))
            self.w_fc12_relu = tf.Variable(tf.truncated_normal([2, releate_node], stddev=0.1))
            self.b_fc12_relu = tf.Variable(tf.constant(0.1, shape=[releate_node]))

            self.h_fc11_relu = tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu
            self.h_fc12_relu = tf.matmul(self.log_v_multi, self.w_fc12_relu) + self.b_fc12_relu

            self.w_fc21_one = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc21_one = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc21_two = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc21_two = tf.Variable(tf.constant(0.1, shape=[1]))
            self.m_one = tf.matmul(self.h_fc11_relu, self.w_fc21_one) + self.b_fc21_one
            self.m_two = tf.matmul(self.h_fc11_relu, self.w_fc21_two) + self.b_fc21_two
            self.m_one = tf.reshape(self.m_one, [-1, self.pointsDecodeTrainNums, 1])
            self.m_two = tf.reshape(self.m_two, [-1, self.pointsDecodeTrainNums, 1])

            self.w_fc22_one = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc22_one = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc22_two = tf.Variable(tf.truncated_normal([releate_node, 1], stddev=0.1))
            self.b_fc22_two = tf.Variable(tf.constant(0.1, shape=[1]))
            self.log_v_one = tf.matmul(self.h_fc12_relu, self.w_fc22_one) + self.b_fc22_one
            self.log_v_two = tf.matmul(self.h_fc12_relu, self.w_fc22_two) + self.b_fc22_two
            self.log_v_one = tf.reshape(self.log_v_one, [-1, self.pointsDecodeTrainNums, 1])
            self.log_v_two = tf.reshape(self.log_v_two, [-1, self.pointsDecodeTrainNums, 1])
            

            # add relu
            # self.w_fc11_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc11_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc12_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc12_relu = tf.Variable(tf.constant(0.1, shape=[1]))

            # # 使用激活函数
            # if using_activate_func:
            #     # self.m_one_relu = tf.nn.relu(tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu)
            #     # self.m_two_relu = tf.nn.relu(tf.matmul(self.m_multi, self.w_fc12_relu) + self.b_fc12_relu)
            #     self.m_one_relu = tf.nn.sigmoid(tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu)
            #     self.m_two_relu = tf.nn.sigmoid(tf.matmul(self.m_multi, self.w_fc12_relu) + self.b_fc12_relu)
            # else:
            #     self.m_one_relu = tf.matmul(self.m_multi, self.w_fc11_relu) + self.b_fc11_relu
            #     self.m_two_relu = tf.matmul(self.m_multi, self.w_fc12_relu) + self.b_fc12_relu
            # self.m_one_relu = tf.reshape(self.m_one_relu, [-1, self.pointsDecodeTrainNums, 1])
            # self.m_two_relu = tf.reshape(self.m_two_relu, [-1, self.pointsDecodeTrainNums, 1])

            # self.w_fc21_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc21_relu = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc22_relu = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc22_relu = tf.Variable(tf.constant(0.1, shape=[1]))

            # if using_activate_func:
            #     # self.log_v_one_relu = tf.nn.relu(tf.matmul(self.log_v_multi, self.w_fc21_relu) + self.b_fc21_relu)
            #     # self.log_v_two_relu = tf.nn.relu(tf.matmul(self.log_v_multi, self.w_fc22_relu) + self.b_fc22_relu)
            #     self.log_v_one_relu = tf.nn.sigmoid(tf.matmul(self.log_v_multi, self.w_fc21_relu) + self.b_fc21_relu)
            #     self.log_v_two_relu = tf.nn.sigmoid(tf.matmul(self.log_v_multi, self.w_fc22_relu) + self.b_fc22_relu)
            # else:
            #     self.log_v_one_relu = tf.matmul(self.log_v_multi, self.w_fc21_relu) + self.b_fc21_relu
            #     self.log_v_two_relu = tf.matmul(self.log_v_multi, self.w_fc22_relu) + self.b_fc22_relu
            # self.log_v_one_relu = tf.reshape(self.log_v_one_relu, [-1, self.pointsDecodeTrainNums, 1])
            # self.log_v_two_relu = tf.reshape(self.log_v_two_relu, [-1, self.pointsDecodeTrainNums, 1])

            # # 最后一层输出
            # self.m_multi_relu = tf.reshape(tf.concat([self.m_one_relu, self.m_two_relu], axis=-1), [-1, 2])  ## 
            # self.log_v_multi_relu = tf.reshape(tf.concat([self.log_v_one_relu, self.log_v_two_relu], axis=-1), [-1, 2])

            # self.w_fc11 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc12 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))

            # self.m_one = tf.matmul(self.m_multi_relu, self.w_fc11) + self.b_fc11
            # self.m_one = tf.reshape(self.m_one, [-1, self.pointsDecodeTrainNums, 1])
            # self.m_two = tf.matmul(self.m_multi_relu, self.w_fc12) + self.b_fc12
            # self.m_two = tf.reshape(self.m_two, [-1, self.pointsDecodeTrainNums, 1])
            # print("self.m_one:", self.m_one)

            # self.w_fc21 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
            # self.w_fc22 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            # self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))

            # self.log_v_one = tf.matmul(self.log_v_multi_relu, self.w_fc21) + self.b_fc21
            # self.log_v_one = tf.reshape(self.log_v_one, [-1, self.pointsDecodeTrainNums, 1])
            # self.log_v_two = tf.matmul(self.log_v_multi_relu, self.w_fc22) + self.b_fc22
            # self.log_v_two = tf.reshape(self.log_v_two, [-1, self.pointsDecodeTrainNums, 1])
            # print("self.log_v_one:", self.log_v_one)

        else:         
            # 最后一层均值输出
            self.w_fc11 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc12 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))

            self.m_one = tf.matmul(self.m_multi, self.w_fc11) + self.b_fc11
            self.m_one = tf.reshape(self.m_one, [-1, 1, 1])

            self.m_two = tf.matmul(self.m_multi, self.w_fc12) + self.b_fc12
            self.m_two = tf.reshape(self.m_two, [-1, 1, 1])



            # 最后一层方差输出
            self.w_fc21 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
            self.w_fc22 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
            self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))

            self.log_v_one = tf.matmul(self.log_v_multi, self.w_fc21) + self.b_fc21
            self.log_v_one = tf.reshape(self.log_v_one, [-1, 1, 1])

            self.log_v_two = tf.matmul(self.log_v_multi, self.w_fc22) + self.b_fc22
            self.log_v_two = tf.reshape(self.log_v_two, [-1, 1, 1])

        if add_task_layer:
            # ------------------------------------------------------------------
            # 在每个输出在后面添加一层
            self.w_fc11_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc11_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.w_fc12_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc12_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.m_one_ = tf.matmul(self.m_one, self.w_fc11_) + self.b_fc11_
            self.m_one_ = tf.reshape(self.m_one_, [-1, self.pointsDecodeTrainNums, 1])
            self.m_two_ = tf.matmul(self.m_two, self.w_fc12_) + self.b_fc12
            self.m_two_ = tf.reshape(self.m_two_, [-1, self.pointsDecodeTrainNums, 1])

            self.w_fc21_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc21_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.w_fc22_ = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
            self.b_fc22_ = tf.Variable(tf.constant(0.001, shape=[1]))
            self.log_v_one_ = tf.matmul(self.log_v_one, self.w_fc21_) + self.b_fc21_
            self.log_v_one_ = tf.reshape(self.log_v_one_, [-1, self.pointsDecodeTrainNums, 1])
            self.log_v_two_ = tf.matmul(self.log_v_two, self.w_fc22_) + self.b_fc22_
            self.log_v_two_ = tf.reshape(self.log_v_two_, [-1, self.pointsDecodeTrainNums, 1])
            # -------------------------------------------------------------

        # 预测值
        if add_task_layer:
            self.sigma_one = 0.1 + 0.9 * tf.nn.softplus(self.log_v_one_)
            self.sigma_two = 0.1 + 0.9 * tf.nn.softplus(self.log_v_two_)
            self.pre_mean_one = tf.reshape(self.m_one_, [1, 1])
            self.pre_sigma_one = tf.reshape(self.sigma_one, [1, 1])
            self.pre_mean_two = tf.reshape(self.m_two_, [1, 1])
            self.pre_sigma_two = tf.reshape(self.sigma_two, [1, 1])
        else:
            self.sigma_one = 0.1 + 0.9 * tf.nn.softplus(self.log_v_one)
            self.sigma_two = 0.1 + 0.9 * tf.nn.softplus(self.log_v_two)
            self.pre_mean_one = tf.reshape(self.m_one, [1, 1])
            self.pre_sigma_one = tf.reshape(self.sigma_one, [1, 1])
            self.pre_mean_two = tf.reshape(self.m_two, [1, 1])
            self.pre_sigma_two = tf.reshape(self.sigma_two, [1, 1])

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        # 任务一参数
        self.param_dict['fc_111w'] = self.w_fc111
        self.param_dict['fc_111b'] = self.b_fc111
        self.param_dict['fc_112w'] = self.w_fc112
        self.param_dict['fc_112b'] = self.b_fc112
        self.param_dict['fc_113w'] = self.w_fc113
        self.param_dict['fc_113b'] = self.b_fc113
        self.param_dict['fc_121w'] = self.w_fc121
        self.param_dict['fc_121b'] = self.b_fc121
        self.param_dict['fc_122w'] = self.w_fc122
        self.param_dict['fc_122b'] = self.b_fc122
        self.param_dict['fc_123w'] = self.w_fc123
        self.param_dict['fc_123b'] = self.b_fc123
        self.param_dict['fc_124w'] = self.w_fc124
        self.param_dict['fc_124b'] = self.b_fc124
        self.param_dict['fc_125w'] = self.w_fc125
        self.param_dict['fc_125b'] = self.b_fc125
        self.param_dict['fc_126w'] = self.w_fc126
        self.param_dict['fc_126b'] = self.b_fc126
        # 任务二参数
        self.param_dict['fc_211w'] = self.w_fc211
        self.param_dict['fc_211b'] = self.b_fc211
        self.param_dict['fc_212w'] = self.w_fc212
        self.param_dict['fc_212b'] = self.b_fc212
        self.param_dict['fc_213w'] = self.w_fc213
        self.param_dict['fc_213b'] = self.b_fc213
        self.param_dict['fc_221w'] = self.w_fc221
        self.param_dict['fc_221b'] = self.b_fc221
        self.param_dict['fc_222w'] = self.w_fc222
        self.param_dict['fc_222b'] = self.b_fc222
        self.param_dict['fc_223w'] = self.w_fc223
        self.param_dict['fc_223b'] = self.b_fc223
        self.param_dict['fc_224w'] = self.w_fc224
        self.param_dict['fc_224b'] = self.b_fc224
        self.param_dict['fc_225w'] = self.w_fc225
        self.param_dict['fc_225b'] = self.b_fc225
        self.param_dict['fc_226w'] = self.w_fc226
        self.param_dict['fc_226b'] = self.b_fc226
           
        # add relu
        if add_layer:
            self.param_dict['fc_11w_relu'] = self.w_fc11_relu
            self.param_dict['fc_11b_relu'] = self.b_fc11_relu
            self.param_dict['fc_12w_relu'] = self.w_fc12_relu
            self.param_dict['fc_12b_relu'] = self.b_fc12_relu

            self.param_dict['fc_21w_one'] = self.w_fc21_one
            self.param_dict['fc_21b_one'] = self.b_fc21_one
            self.param_dict['fc_21w_two'] = self.w_fc21_two
            self.param_dict['fc_21b_two'] = self.b_fc21_two

            self.param_dict['fc_22w_one'] = self.w_fc22_one
            self.param_dict['fc_22b_one'] = self.b_fc22_one
            self.param_dict['fc_22w_two'] = self.w_fc22_two
            self.param_dict['fc_22b_two'] = self.b_fc22_two

        #  # 最后连接层
        self.param_dict['fc_11w'] = self.w_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.w_fc12
        self.param_dict['fc_12b'] = self.b_fc12
        self.param_dict['fc_21w'] = self.w_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.w_fc22
        self.param_dict['fc_22b'] = self.b_fc22

        if add_task_layer:
            self.param_dict['fc_11w_'] = self.w_fc11_
            self.param_dict['fc_11b_'] = self.b_fc11_
            self.param_dict['fc_12w_'] = self.w_fc12_
            self.param_dict['fc_12b_'] = self.b_fc12_

            self.param_dict['fc_21w_'] = self.w_fc21_
            self.param_dict['fc_21b_'] = self.b_fc21_
            self.param_dict['fc_22w_'] = self.w_fc22_
            self.param_dict['fc_22b_'] = self.b_fc22_


        # 读取参数
        ss = tf.train.Saver(self.param_dict)
        ss.restore(self.sess, './cnp_model/{}/cnp_model'.format(self.load_dir))

    def set_predict_point_1(self, pred_x, task): ## pred_x 是一个8维的列表
        self.task = task
        self.pred_x = pred_x

   
    ## 做预测是，所有CNP的预测输入都是一样的
    def set_predict_point(self, pre_x1, pre_x2, flag):
        self.flag = flag
        self.pre_x11 = pre_x1  ## 输入一样
        self.pre_x12 = pre_x2  ##
        self.pre_x21 = pre_x1
        self.pre_x22 = pre_x2
        self.pre_x31 = pre_x1
        self.pre_x32 = pre_x2
        
    def cnp_predict_model_1(self, observe_point_one, observe_point_two, pred_x):
        Y_one = np.zeros([1, 1, 1])
        Y_two = np.zeros([1, 1, 1])
        # 利用模型预测，每个CNP模型需要预测的x必须是一样的
        if self.task == 1:
            predict_mean  = self.pre_mean_one.eval(feed_dict ={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x})
            predict_sigma = self.pre_sigma_one.eval(feed_dict={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x})
        elif self.task == 2:
            predict_mean  = self.pre_mean_two.eval(feed_dict ={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x})
            predict_sigma = self.pre_sigma_two.eval(feed_dict={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x})
        else:
            print('The flag must be 0 or 1！')
        return predict_mean, predict_sigma

    def close_sess(self):
        tf.reset_default_graph()
        self.sess.close()
