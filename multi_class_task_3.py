# -*- coding:utf-8 -*-
# Author： DYF
# multi_class_task_3: 3个任务


# coding:utf-8
import numpy as np
import tensorflow as tf
import os

## 20190505 开始进行ParEGO_MTCNP
## 目前是两个任务

class TrainNet:
    CON_LAYERS = 256
    TRAIN_STEP = 1000

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
        
         # 任务三---------------------------------------
        self.x_three = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.test_dimension+1])
        self.y_three = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, 1])
        # 预测x的占位符,修改。
        self.x_t_three = tf.placeholder(tf.float32, [None, self.pointsDecodeTrainNums, self.test_dimension])
        # 输入格式变化
        self.input_x_three = tf.reshape(self.x_three, [-1, self.test_dimension+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        self.w_fc311 = tf.Variable(tf.truncated_normal([self.test_dimension+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc311 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc311 = tf.nn.relu(tf.matmul(self.input_x_three, self.w_fc311) + self.b_fc311)

        # 编码器第二层
        self.w_fc312 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc312 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc312 = tf.nn.relu(tf.matmul(self.h_fc311, self.w_fc312) + self.b_fc312)

        # 编码器第三层
        self.w_fc313 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc313 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc313 = tf.matmul(self.h_fc312, self.w_fc313) + self.b_fc313
        self.h_fc313 = tf.reshape(self.h_fc313, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc313 = tf.reduce_mean(self.h_fc313, axis=1)

        # 编码器输出数据处理
        self.representation_three = tf.tile(tf.expand_dims(self.h_fc313, axis=1), [1, self.pointsDecodeTrainNums, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec_three = tf.concat([self.representation_three, self.x_t_three], axis=-1)
        self.dec_three = tf.reshape(self.dec_three, [-1, TrainNet.CON_LAYERS + self.test_dimension])

        # 解码器第一层
        self.w_fc321 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.test_dimension, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc321 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc321 = tf.nn.relu(tf.matmul(self.dec_three, self.w_fc321) + self.b_fc321)

        # 解码器第二层
        self.w_fc322 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc322 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc322 = tf.nn.relu(tf.matmul(self.h_fc321, self.w_fc322) + self.b_fc322)

        # 解码器第三层
        self.w_fc323 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc323 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc323 = tf.nn.relu(tf.matmul(self.h_fc322, self.w_fc323) + self.b_fc323)

        # 解码器第四层
        self.w_fc324 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc324 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc324 = tf.nn.relu(tf.matmul(self.h_fc323, self.w_fc324) + self.b_fc324)

        # 解码器第五层方差
        self.w_fc325 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc325 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_three = tf.matmul(self.h_fc324, self.w_fc325) + self.b_fc325
        self.log_v_three = tf.reshape(self.log_v_three, [-1, self.pointsDecodeTrainNums, 1])

        # 解码器第五层均值
        self.w_fc326 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc326 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_three = tf.matmul(self.h_fc324, self.w_fc326) + self.b_fc326
        self.m_three = tf.reshape(self.m_three, [-1, self.pointsDecodeTrainNums, 1])
        
        

        
        ## 将多个任务的均值和方差，进行全连接输出   3--->2
        self.m_multi = tf.reshape(tf.concat([self.m_one, self.m_two, self.m_three], axis=-1), [-1, 3])  ## 
        self.log_v_multi = tf.reshape(tf.concat([self.log_v_one, self.log_v_two, self.log_v_three], axis=-1), [-1, 3])
        # 最后一层均值输出
        self.w_fc11 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc12 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc13 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc13 = tf.Variable(tf.constant(0.1, shape=[1]))



        self.m_one = tf.matmul(self.m_multi, self.w_fc11) + self.b_fc11
        self.m_one = tf.reshape(self.m_one, [-1, self.pointsDecodeTrainNums, 1])
        self.m_two = tf.matmul(self.m_multi, self.w_fc12) + self.b_fc12
        self.m_two = tf.reshape(self.m_two, [-1, self.pointsDecodeTrainNums, 1])
        self.m_three = tf.matmul(self.m_multi, self.w_fc13) + self.b_fc13
        self.m_three = tf.reshape(self.m_three, [-1, self.pointsDecodeTrainNums, 1])

        # 最后一层方差输出
        self.w_fc21 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc22 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc23 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc23 = tf.Variable(tf.constant(0.1, shape=[1]))

        self.log_v_one = tf.matmul(self.log_v_multi, self.w_fc21) + self.b_fc21
        self.log_v_one = tf.reshape(self.log_v_one, [-1, self.pointsDecodeTrainNums, 1])
        self.log_v_two = tf.matmul(self.log_v_multi, self.w_fc22) + self.b_fc22
        self.log_v_two = tf.reshape(self.log_v_two, [-1, self.pointsDecodeTrainNums, 1])
        self.log_v_three = tf.matmul(self.log_v_multi, self.w_fc23) + self.b_fc23
        self.log_v_three = tf.reshape(self.log_v_three, [-1, self.pointsDecodeTrainNums, 1])


        # 损失函数
        self.sigma_three = 0.1 + 0.9 * tf.nn.softplus(self.log_v_three)
        self.dist_three = tf.contrib.distributions.MultivariateNormalDiag(loc=self.m_three, scale_diag=self.sigma_three)
        self.log_p_three = self.dist_three.log_prob(self.y_three)
        self.loss_three = -tf.reduce_mean(self.log_p_three)
         
        self.sigma_two = 0.1 + 0.9 * tf.nn.softplus(self.log_v_two)
        self.dist_two = tf.contrib.distributions.MultivariateNormalDiag(loc=self.m_two, scale_diag=self.sigma_two)
        self.log_p_two = self.dist_two.log_prob(self.y_two)
        self.loss_two = -tf.reduce_mean(self.log_p_two)
       
        self.sigma_one = 0.1 + 0.9 * tf.nn.softplus(self.log_v_one)
        self.dist_one = tf.contrib.distributions.MultivariateNormalDiag(loc=self.m_one, scale_diag=self.sigma_one)
        self.log_p_one = self.dist_one.log_prob(self.y_one)
        self.loss_one = -tf.reduce_mean(self.log_p_one)

        self.loss = self.loss_one + self.loss_two + self.loss_three

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_step = self.optimizer.minimize(self.loss)
        tf.global_variables_initializer().run()
        
     
    def cnp_train_model_1(self, observe_point_one, total_point_x_one, total_point_y_one, observe_point_two, total_point_x_two, total_point_y_two, observe_point_three, total_point_x_three, total_point_y_three):
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
        
        # 任务三参数
        self.param_dict['fc_311w'] = self.w_fc311
        self.param_dict['fc_311b'] = self.b_fc311
        self.param_dict['fc_312w'] = self.w_fc312
        self.param_dict['fc_312b'] = self.b_fc312
        self.param_dict['fc_313w'] = self.w_fc313
        self.param_dict['fc_313b'] = self.b_fc313
        self.param_dict['fc_321w'] = self.w_fc321
        self.param_dict['fc_321b'] = self.b_fc321
        self.param_dict['fc_322w'] = self.w_fc322
        self.param_dict['fc_322b'] = self.b_fc322
        self.param_dict['fc_323w'] = self.w_fc323
        self.param_dict['fc_323b'] = self.b_fc323
        self.param_dict['fc_324w'] = self.w_fc324
        self.param_dict['fc_324b'] = self.b_fc324
        self.param_dict['fc_325w'] = self.w_fc325
        self.param_dict['fc_325b'] = self.b_fc325
        self.param_dict['fc_326w'] = self.w_fc326
        self.param_dict['fc_326b'] = self.b_fc326
        
        # 最后连接层
        self.param_dict['fc_11w'] = self.w_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.w_fc12
        self.param_dict['fc_12b'] = self.b_fc12
        self.param_dict['fc_13w'] = self.w_fc13
        self.param_dict['fc_13b'] = self.b_fc13        

        self.param_dict['fc_21w'] = self.w_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.w_fc22
        self.param_dict['fc_22b'] = self.b_fc22
        self.param_dict['fc_23w'] = self.w_fc23
        self.param_dict['fc_23b'] = self.b_fc23
        saver = tf.train.Saver(self.param_dict)
        
        # loss_value_history = []
        # 模型训练
        for i in range(TrainNet.TRAIN_STEP):
            self.train_step.run(feed_dict={self.x_one: observe_point_one, self.y_one: total_point_y_one, self.x_t_one: total_point_x_one,
                                           self.x_two: observe_point_two, self.y_two: total_point_y_two, self.x_t_two: total_point_x_two,
                                           self.x_three: observe_point_three, self.y_three: total_point_y_three, self.x_t_three: total_point_x_three})
            ##loss_value = loss_value + self.loss.eval(feed_dict={self.x_one: observe_point_one, self.y_one: total_point_y_one, 
            ##                                                    self.x_t_one:total_point_x_one,                                                                   ##                                                      self.x_two: observe_point_two, self.y_two: total_point_y_two, 
            ##                                                    self.x_t_two: total_point_x_two}).mean()
            loss_value = self.loss.eval(feed_dict={self.x_one: observe_point_one, self.y_one: total_point_y_one, self.x_t_one: total_point_x_one,
                                        self.x_two: observe_point_two, self.y_two: total_point_y_two, self.x_t_two: total_point_x_two,
                                        self.x_three: observe_point_three, self.y_three: total_point_y_three, self.x_t_three: total_point_x_three}).mean()
            # 5000次打印一次总loss的平均值
            if i % (TrainNet.TRAIN_STEP - 1) == 0:
                if i == 0:
                    continue
                ##print("error:第{}次 {}".format(i + 1, loss_value / (3 * TrainNet.TRAIN_STEP)))
                print("error:第{}次 {}".format(i + 1, loss_value))
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


    
    ## 输入观察点（x,y）,total_point_x, total_point_y
    def cnp_train_model(self, x11, x12, x21, x22, x31, x32, code_1y, code_2y, code_3y,
                        dex11, dex12, decode_1y, dex21, dex22, decode_2y, dex31, dex32, decode_3y):
        self.x11 = x11
        self.x12 = x12
        self.code_1y = code_1y
        self.decode_1y = decode_1y
        self.dex11 = dex11
        self.dex12 = dex12

        self.x21 = x21
        self.x22 = x22
        self.code_2y = code_2y
        self.decode_2y = decode_2y
        self.dex21 = dex21
        self.dex22 = dex22

        self.x31 = x31
        self.x32 = x32
        self.code_3y = code_3y
        self.decode_3y = decode_3y
        self.dex31 = dex31
        self.dex32 = dex32

        # 产生初始数据
        X_one = np.zeros([1, self.pointsCodeTrainNums, self.test_dimension+1])
        X_two = np.zeros([1, self.pointsCodeTrainNums, self.test_dimension+1])
        X_three = np.zeros([1, self.pointsCodeTrainNums, self.test_dimension+1])
        # 预测、解码器的点初始化
        predict_X_one = np.zeros([1, self.pointsDecodeTrainNums, self.test_dimension])
        predict_X_two = np.zeros([1, self.pointsDecodeTrainNums, self.test_dimension])
        predict_X_three = np.zeros([1, self.pointsDecodeTrainNums, self.test_dimension])
        # 真实值初始化
        Y_one = np.zeros([1, self.pointsDecodeTrainNums, 1])
        Y_two = np.zeros([1, self.pointsDecodeTrainNums, 1])
        Y_three = np.zeros([1, self.pointsDecodeTrainNums, 1])
        loss_value = 0

        # 模型训练数据
        # 任务一編碼器的输入
        for j, value in enumerate(self.x11):
            X_one[0, j, 0] = value
        for k, value in enumerate(self.x12):
            X_one[0, k, 1] = value
        for l, value in enumerate(self.code_1y):
            X_one[0, l, 2] = (value - min(self.decode_1y)) / (max(self.decode_1y) - min(self.decode_1y))
        # 解碼器輸入
        for j, value in enumerate(self.dex11):
            predict_X_one[0, j, 0] = value
        for k, value in enumerate(self.dex12):
            predict_X_one[0, k, 1] = value
        for l, value in enumerate(self.decode_1y):
            Y_one[0, l, 0] = (value - min(self.decode_1y)) / (max(self.decode_1y) - min(self.decode_1y))

        # 任务二編碼器的输入
        for j, value in enumerate(self.x21):
            X_two[0, j, 0] = value
        for k, value in enumerate(self.x22):
            X_two[0, k, 1] = value
        for l, value in enumerate(self.code_2y):
            X_two[0, l, 2] = (value - min(self.decode_2y)) / (max(self.decode_2y) - min(self.decode_2y))
        # 解碼器輸入
        for j, value in enumerate(self.dex21):
            predict_X_two[0, j, 0] = value
        for k, value in enumerate(self.dex22):
            predict_X_two[0, k, 1] = value
        for l, value in enumerate(self.decode_2y):
            Y_two[0, l, 0] = (value - min(self.decode_2y)) / (max(self.decode_2y) - min(self.decode_2y))

        # 任务三編碼器的输入
        for j, value in enumerate(self.x31):
            X_three[0, j, 0] = value
        for k, value in enumerate(self.x32):
            X_three[0, k, 1] = value
        for l, value in enumerate(self.code_3y):
            X_three[0, l, 2] = (value - min(self.decode_3y)) / (max(self.decode_3y) - min(self.decode_3y))
        # 解碼器輸入
        for j, value in enumerate(self.dex31):
            predict_X_three[0, j, 0] = value
        for k, value in enumerate(self.dex32):
            predict_X_three[0, k, 1] = value
        for l, value in enumerate(self.decode_3y):
            Y_three[0, l, 0] = (value - min(self.decode_3y)) / (max(self.decode_3y) - min(self.decode_3y))

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
        # 任务三参数
        self.param_dict['fc_311w'] = self.w_fc311
        self.param_dict['fc_311b'] = self.b_fc311
        self.param_dict['fc_312w'] = self.w_fc312
        self.param_dict['fc_312b'] = self.b_fc312
        self.param_dict['fc_313w'] = self.w_fc313
        self.param_dict['fc_313b'] = self.b_fc313
        self.param_dict['fc_321w'] = self.w_fc321
        self.param_dict['fc_321b'] = self.b_fc321
        self.param_dict['fc_322w'] = self.w_fc322
        self.param_dict['fc_322b'] = self.b_fc322
        self.param_dict['fc_323w'] = self.w_fc323
        self.param_dict['fc_323b'] = self.b_fc323
        self.param_dict['fc_324w'] = self.w_fc324
        self.param_dict['fc_324b'] = self.b_fc324
        self.param_dict['fc_325w'] = self.w_fc325
        self.param_dict['fc_325b'] = self.b_fc325
        self.param_dict['fc_326w'] = self.w_fc326
        self.param_dict['fc_326b'] = self.b_fc326
        # 最后连接层
        self.param_dict['fc_11w'] = self.w_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.w_fc12
        self.param_dict['fc_12b'] = self.b_fc12
        self.param_dict['fc_13w'] = self.w_fc13
        self.param_dict['fc_13b'] = self.b_fc13
        self.param_dict['fc_21w'] = self.w_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.w_fc22
        self.param_dict['fc_22b'] = self.b_fc22
        self.param_dict['fc_23w'] = self.w_fc23
        self.param_dict['fc_23b'] = self.b_fc23
        saver = tf.train.Saver(self.param_dict)

        # loss_value_history = []
        # 模型训练
        for i in range(TrainNet.TRAIN_STEP):
            self.train_step.run(feed_dict={self.x_one: X_one, self.x_two: X_two, self.x_three: X_three,
                                           self.y_one: Y_one, self.y_two: Y_two, self.y_three: Y_three,
                                           self.x_t_one: predict_X_one, self.x_t_two: predict_X_two,
                                           self.x_t_three: predict_X_three})
            loss_value = loss_value + self.loss.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
                                                                self.x_three: X_three, self.y_one: Y_one,
                                                                self.y_two: Y_two, self.y_three: Y_three,
                                                                self.x_t_one: predict_X_one,
                                                                self.x_t_two: predict_X_two,
                                                                self.x_t_three: predict_X_three}).mean()
            # 5000次打印一次总loss的平均值
            if i % (TrainNet.TRAIN_STEP - 1) == 0:
                if i == 0:
                    continue
                print("error:第{}次 {}".format(i + 1, loss_value / (3 * TrainNet.TRAIN_STEP)))
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
    
    def set_train_point_1(self,code_num, observe_point_one, total_point_y_one, observe_point_two, total_point_y_two, observe_point_three, total_point_y_three):
        self.pointsCodeTrainNums = code_num
        self.observe_point_one = observe_point_one
        self.total_point_y_one = total_point_y_one
        self.observe_point_two = observe_point_two
        self.total_point_y_two = total_point_y_two
        self.observe_point_three = observe_point_three
        self.total_point_y_three = total_point_y_three
        
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
        
        
        
        # 任务三---------------------------------------
        self.x_three = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.test_dimension+1])
        self.y_three = tf.placeholder(tf.float32, [None, 1, 1])
        # 预测x的占位符,修改。
        self.x_t_three = tf.placeholder(tf.float32, [None, 1, self.test_dimension])
        # 输入格式变化
        self.input_x_three = tf.reshape(self.x_three, [-1, self.test_dimension+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        self.w_fc311 = tf.Variable(tf.truncated_normal([self.test_dimension+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc311 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc311 = tf.nn.relu(tf.matmul(self.input_x_three, self.w_fc311) + self.b_fc311)

        # 编码器第二层
        self.w_fc312 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc312 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc312 = tf.nn.relu(tf.matmul(self.h_fc311, self.w_fc312) + self.b_fc312)

        # 编码器第三层
        self.w_fc313 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc313 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc313 = tf.matmul(self.h_fc312, self.w_fc313) + self.b_fc313
        self.h_fc313 = tf.reshape(self.h_fc313, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc313 = tf.reduce_mean(self.h_fc313, axis=1)

        # 编码器输出数据处理
        self.representation_three = tf.tile(tf.expand_dims(self.h_fc313, axis=1), [1, 1, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec_three = tf.concat([self.representation_three, self.x_t_three], axis=-1)
        self.dec_three = tf.reshape(self.dec_three, [-1, TrainNet.CON_LAYERS + self.test_dimension])

        # 解码器第一层
        self.w_fc321 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.test_dimension, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc321 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc321 = tf.nn.relu(tf.matmul(self.dec_three, self.w_fc321) + self.b_fc321)

        # 解码器第二层
        self.w_fc322 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc322 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc322 = tf.nn.relu(tf.matmul(self.h_fc321, self.w_fc322) + self.b_fc322)

        # 解码器第三层
        self.w_fc323 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc323 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc323 = tf.nn.relu(tf.matmul(self.h_fc322, self.w_fc323) + self.b_fc323)

        # 解码器第四层
        self.w_fc324 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc324 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc324 = tf.nn.relu(tf.matmul(self.h_fc323, self.w_fc324) + self.b_fc324)

        # 解码器第五层方差
        self.w_fc325 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc325 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_three = tf.matmul(self.h_fc324, self.w_fc325) + self.b_fc325
        self.log_v_three = tf.reshape(self.log_v_three, [-1, 1, 1])

        # 解码器第五层均值
        self.w_fc326 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc326 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_three = tf.matmul(self.h_fc324, self.w_fc326) + self.b_fc326
        self.m_three = tf.reshape(self.m_three, [-1, 1, 1])
        
        self.m_multi = tf.reshape(tf.concat([self.m_one, self.m_two, self.m_three], axis=-1), [-1, 3])
        #self.log_v_multi = tf.reshape(tf.concat([self.log_v_one, self.log_v_two, self.log_v_two], axis=-1), [-1, 3])  ## 201905181505 发现错误
        self.log_v_multi = tf.reshape(tf.concat([self.log_v_one, self.log_v_two, self.log_v_three], axis=-1), [-1, 3])
        # 最后一层均值输出
        self.w_fc11 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc12 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc13 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc13 = tf.Variable(tf.constant(0.1, shape=[1]))

        self.m_one = tf.matmul(self.m_multi, self.w_fc11) + self.b_fc11
        self.m_one = tf.reshape(self.m_one, [-1, 1, 1])

        self.m_two = tf.matmul(self.m_multi, self.w_fc12) + self.b_fc12
        self.m_two = tf.reshape(self.m_two, [-1, 1, 1])
        
        self.m_three = tf.matmul(self.m_multi, self.w_fc13) + self.b_fc13
        self.m_three = tf.reshape(self.m_three, [-1, 1, 1])



        # 最后一层方差输出
        self.w_fc21 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc22 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc23 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc23 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_one = tf.matmul(self.log_v_multi, self.w_fc21) + self.b_fc21
        self.log_v_one = tf.reshape(self.log_v_one, [-1, 1, 1])

        self.log_v_two = tf.matmul(self.log_v_multi, self.w_fc22) + self.b_fc22
        self.log_v_two = tf.reshape(self.log_v_two, [-1, 1, 1])
        
        self.log_v_three = tf.matmul(self.log_v_multi, self.w_fc23) + self.b_fc23
        self.log_v_three = tf.reshape(self.log_v_three, [-1, 1, 1])



        # 预测值
        self.sigma_three = 0.1 + 0.9 * tf.nn.softplus(self.log_v_three)
        self.sigma_two = 0.1 + 0.9 * tf.nn.softplus(self.log_v_two)
        self.sigma_one = 0.1 + 0.9 * tf.nn.softplus(self.log_v_one)
        self.pre_mean_one = tf.reshape(self.m_one, [1, 1])
        self.pre_sigma_one = tf.reshape(self.sigma_one, [1, 1])
        self.pre_mean_two = tf.reshape(self.m_two, [1, 1])
        self.pre_sigma_two = tf.reshape(self.sigma_two, [1, 1])
        self.pre_mean_three = tf.reshape(self.m_three, [1, 1])
        self.pre_sigma_three = tf.reshape(self.sigma_three, [1, 1])

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
        # 任务三参数
        self.param_dict['fc_311w'] = self.w_fc311
        self.param_dict['fc_311b'] = self.b_fc311
        self.param_dict['fc_312w'] = self.w_fc312
        self.param_dict['fc_312b'] = self.b_fc312
        self.param_dict['fc_313w'] = self.w_fc313
        self.param_dict['fc_313b'] = self.b_fc313
        self.param_dict['fc_321w'] = self.w_fc321
        self.param_dict['fc_321b'] = self.b_fc321
        self.param_dict['fc_322w'] = self.w_fc322
        self.param_dict['fc_322b'] = self.b_fc322
        self.param_dict['fc_323w'] = self.w_fc323
        self.param_dict['fc_323b'] = self.b_fc323
        self.param_dict['fc_324w'] = self.w_fc324
        self.param_dict['fc_324b'] = self.b_fc324
        self.param_dict['fc_325w'] = self.w_fc325
        self.param_dict['fc_325b'] = self.b_fc325
        self.param_dict['fc_326w'] = self.w_fc326
        self.param_dict['fc_326b'] = self.b_fc326
        
        
         # 最后连接层
        self.param_dict['fc_11w'] = self.w_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.w_fc12
        self.param_dict['fc_12b'] = self.b_fc12
        self.param_dict['fc_13w'] = self.w_fc13
        self.param_dict['fc_13b'] = self.b_fc13
        self.param_dict['fc_21w'] = self.w_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.w_fc22
        self.param_dict['fc_22b'] = self.b_fc22
        self.param_dict['fc_23w'] = self.w_fc23
        self.param_dict['fc_23b'] = self.b_fc23
        # 读取参数
        ss = tf.train.Saver(self.param_dict)
        ss.restore(self.sess, './cnp_model/{}/cnp_model'.format(self.load_dir))

        
    def set_predict_point_1(self, pred_x, task): ## pred_x 是一个8维的列表
        self.task = task
        self.pred_x = pred_x
        
    
    def set_train_point(self, code_num, x11, x12, code_1y, x21, x22, x31, x32, decode_1y, code_2y,
                        decode_2y, code_3y, decode_3y):
        self.pointsCodeTrainNums = code_num
        ## 下面self.x11 貌似在该函数里面，没有使用
        self.x11 = x11
        self.x12 = x12
        self.code_1y = code_1y
        self.decode_1y = decode_1y

        self.x21 = x21
        self.x22 = x22
        self.code_2y = code_2y
        self.decode_2y = decode_2y

        self.x31 = x31
        self.x32 = x32
        self.code_3y = code_3y
        self.decode_3y = decode_3y

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

        # 任务三---------------------------------------
        self.x_three = tf.placeholder(tf.float32, [None, self.pointsCodeTrainNums, self.test_dimension+1])
        self.y_three = tf.placeholder(tf.float32, [None, 1, 1])
        # 预测x的占位符,修改。
        self.x_t_three = tf.placeholder(tf.float32, [None, 1, self.test_dimension])
        # 输入格式变化
        self.input_x_three = tf.reshape(self.x_three, [-1, self.test_dimension+1])

        # 编码器4第一层,全连接大小可变，具体需要测
        self.w_fc311 = tf.Variable(tf.truncated_normal([self.test_dimension+1, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc311 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc311 = tf.nn.relu(tf.matmul(self.input_x_three, self.w_fc311) + self.b_fc311)

        # 编码器第二层
        self.w_fc312 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc312 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc312 = tf.nn.relu(tf.matmul(self.h_fc311, self.w_fc312) + self.b_fc312)

        # 编码器第三层
        self.w_fc313 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc313 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc313 = tf.matmul(self.h_fc312, self.w_fc313) + self.b_fc313
        self.h_fc313 = tf.reshape(self.h_fc313, [-1, self.pointsCodeTrainNums, TrainNet.CON_LAYERS])
        self.h_fc313 = tf.reduce_mean(self.h_fc313, axis=1)

        # 编码器输出数据处理
        self.representation_three = tf.tile(tf.expand_dims(self.h_fc313, axis=1), [1, 1, 1])
        # representation = tf.reshape(h_fc13, [-1, 1, TrainNet.CON_LAYERS])
        self.dec_three = tf.concat([self.representation_three, self.x_t_three], axis=-1)
        self.dec_three = tf.reshape(self.dec_three, [-1, TrainNet.CON_LAYERS + self.test_dimension])

        # 解码器第一层
        self.w_fc321 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS + self.test_dimension, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc321 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc321 = tf.nn.relu(tf.matmul(self.dec_three, self.w_fc321) + self.b_fc321)

        # 解码器第二层
        self.w_fc322 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc322 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc322 = tf.nn.relu(tf.matmul(self.h_fc321, self.w_fc322) + self.b_fc322)

        # 解码器第三层
        self.w_fc323 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc323 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc323 = tf.nn.relu(tf.matmul(self.h_fc322, self.w_fc323) + self.b_fc323)

        # 解码器第四层
        self.w_fc324 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, TrainNet.CON_LAYERS], stddev=0.1))
        self.b_fc324 = tf.Variable(tf.constant(0.1, shape=[TrainNet.CON_LAYERS]))
        self.h_fc324 = tf.nn.relu(tf.matmul(self.h_fc323, self.w_fc324) + self.b_fc324)

        # 解码器第五层方差
        self.w_fc325 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc325 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_three = tf.matmul(self.h_fc324, self.w_fc325) + self.b_fc325
        self.log_v_three = tf.reshape(self.log_v_three, [-1, 1, 1])

        # 解码器第五层均值
        self.w_fc326 = tf.Variable(tf.truncated_normal([TrainNet.CON_LAYERS, 1], stddev=0.1))
        self.b_fc326 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_three = tf.matmul(self.h_fc324, self.w_fc326) + self.b_fc326
        self.m_three = tf.reshape(self.m_three, [-1, 1, 1])

        self.m_multi = tf.reshape(tf.concat([self.m_one, self.m_two, self.m_three], axis=-1), [-1, 3])
        self.log_v_multi = tf.reshape(tf.concat([self.log_v_one, self.log_v_two, self.log_v_three], axis=-1), [-1, 3])

        # 最后一层均值输出
        self.w_fc11 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc11 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc12 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc12 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc13 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc13 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.m_one = tf.matmul(self.m_multi, self.w_fc11) + self.b_fc11
        self.m_one = tf.reshape(self.m_one, [-1, 1, 1])

        self.m_two = tf.matmul(self.m_multi, self.w_fc12) + self.b_fc12
        self.m_two = tf.reshape(self.m_two, [-1, 1, 1])

        self.m_three = tf.matmul(self.m_multi, self.w_fc13) + self.b_fc13
        self.m_three = tf.reshape(self.m_three, [-1, 1, 1])

        # 最后一层方差输出
        self.w_fc21 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc21 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc22 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc22 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.w_fc23 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.1))
        self.b_fc23 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.log_v_one = tf.matmul(self.log_v_multi, self.w_fc21) + self.b_fc21
        self.log_v_one = tf.reshape(self.log_v_one, [-1, 1, 1])

        self.log_v_two = tf.matmul(self.log_v_multi, self.w_fc22) + self.b_fc22
        self.log_v_two = tf.reshape(self.log_v_two, [-1, 1, 1])

        self.log_v_three = tf.matmul(self.log_v_multi, self.w_fc23) + self.b_fc23
        self.log_v_three = tf.reshape(self.log_v_three, [-1, 1, 1])

        # 预测值
        self.sigma_three = 0.1 + 0.9 * tf.nn.softplus(self.log_v_three)
        self.sigma_two = 0.1 + 0.9 * tf.nn.softplus(self.log_v_two)
        self.sigma_one = 0.1 + 0.9 * tf.nn.softplus(self.log_v_one)
        
        self.pre_mean_one = tf.reshape(self.m_one, [1, 1])
        self.pre_sigma_one = tf.reshape(self.sigma_one, [1, 1])
        self.pre_mean_two = tf.reshape(self.m_two, [1, 1])
        self.pre_sigma_two = tf.reshape(self.sigma_two, [1, 1])
        self.pre_mean_three = tf.reshape(self.m_three, [1, 1])
        self.pre_sigma_three = tf.reshape(self.sigma_three, [1, 1])

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
        # 任务三参数
        self.param_dict['fc_311w'] = self.w_fc311
        self.param_dict['fc_311b'] = self.b_fc311
        self.param_dict['fc_312w'] = self.w_fc312
        self.param_dict['fc_312b'] = self.b_fc312
        self.param_dict['fc_313w'] = self.w_fc313
        self.param_dict['fc_313b'] = self.b_fc313
        self.param_dict['fc_321w'] = self.w_fc321
        self.param_dict['fc_321b'] = self.b_fc321
        self.param_dict['fc_322w'] = self.w_fc322
        self.param_dict['fc_322b'] = self.b_fc322
        self.param_dict['fc_323w'] = self.w_fc323
        self.param_dict['fc_323b'] = self.b_fc323
        self.param_dict['fc_324w'] = self.w_fc324
        self.param_dict['fc_324b'] = self.b_fc324
        self.param_dict['fc_325w'] = self.w_fc325
        self.param_dict['fc_325b'] = self.b_fc325
        self.param_dict['fc_326w'] = self.w_fc326
        self.param_dict['fc_326b'] = self.b_fc326
        # 最后连接层
        self.param_dict['fc_11w'] = self.w_fc11
        self.param_dict['fc_11b'] = self.b_fc11
        self.param_dict['fc_12w'] = self.w_fc12
        self.param_dict['fc_12b'] = self.b_fc12
        self.param_dict['fc_13w'] = self.w_fc13
        self.param_dict['fc_13b'] = self.b_fc13
        self.param_dict['fc_21w'] = self.w_fc21
        self.param_dict['fc_21b'] = self.b_fc21
        self.param_dict['fc_22w'] = self.w_fc22
        self.param_dict['fc_22b'] = self.b_fc22
        self.param_dict['fc_23w'] = self.w_fc23
        self.param_dict['fc_23b'] = self.b_fc23
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
        
    def cnp_predict_model_1(self, observe_point_one, observe_point_two, observe_point_three, pred_x):
        Y_one = np.zeros([1, 1, 1])
        Y_two = np.zeros([1, 1, 1])
        Y_three = np.zeros([1, 1, 1])
        # 利用模型预测，每个CNP模型需要预测的x必须是一样的
        if self.task == 1:
            predict_mean  = self.pre_mean_one.eval(feed_dict ={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x,
                                                               self.x_three: observe_point_three, self.y_three: Y_three, self.x_t_three: pred_x})
            predict_sigma = self.pre_sigma_one.eval(feed_dict={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x,
                                                               self.x_three: observe_point_three, self.y_three: Y_three, self.x_t_three: pred_x})
        elif self.task == 2:
            predict_mean  = self.pre_mean_two.eval(feed_dict ={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x,
                                                               self.x_three: observe_point_three, self.y_three: Y_three, self.x_t_three: pred_x})
            predict_sigma = self.pre_sigma_two.eval(feed_dict={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x,
                                                               self.x_three: observe_point_three, self.y_three: Y_three, self.x_t_three: pred_x})
        elif self.task == 3:
            predict_mean  = self.pre_mean_three.eval(feed_dict ={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x,
                                                               self.x_three: observe_point_three, self.y_three: Y_three, self.x_t_three: pred_x})
            predict_sigma = self.pre_sigma_three.eval(feed_dict={self.x_one: observe_point_one, self.y_one: Y_one, self.x_t_one: pred_x,
                                                               self.x_two: observe_point_two, self.y_two: Y_two, self.x_t_two: pred_x,
                                                               self.x_three: observe_point_three, self.y_three: Y_three, self.x_t_three: pred_x})
        else:
            print('The flag must be 2 or 3！')
        return predict_mean, predict_sigma
            

    def cnp_predict_model(self):
        # 網絡部分
        # 产生初始数据
        X_one = np.zeros([1, self.pointsCodeTrainNums, self.test_dimension+1])
        X_two = np.zeros([1, self.pointsCodeTrainNums, self.test_dimension+1])
        X_three = np.zeros([1, self.pointsCodeTrainNums, self.test_dimension+1])
        # 预测、解码器的点初始化
        predict_X_one = np.zeros([1, 1, self.test_dimension])
        predict_X_two = np.zeros([1, 1, self.test_dimension])
        predict_X_three = np.zeros([1, 1, self.test_dimension])
        # 真实值初始化
        Y_one = np.zeros([1, 1, 1])
        Y_two = np.zeros([1, 1, 1])
        Y_three = np.zeros([1, 1, 1])

        # 模型训练
        # 任务一编码器的输入（参考点）
        for j, value in enumerate(self.x11):
            X_one[0, j, 0] = value
        for k, value in enumerate(self.x12):
            X_one[0, k, 1] = value
        for l, value in enumerate(self.code_1y):
            X_one[0, l, 2] = (value - min(self.decode_1y)) / (max(self.decode_1y) - min(self.decode_1y))
        # 解碼器輸入
        predict_X_one[0, 0, 0] = self.pre_x11
        predict_X_one[0, 0, 1] = self.pre_x12

        # 任务二編碼器的输入
        for j, value in enumerate(self.x21):
            X_two[0, j, 0] = value
        for k, value in enumerate(self.x22):
            X_two[0, k, 1] = value
        for l, value in enumerate(self.code_2y):
            X_two[0, l, 2] = (value - min(self.decode_2y)) / (max(self.decode_2y) - min(self.decode_2y))
        # 解碼器輸入
        predict_X_two[0, 0, 0] = self.pre_x21
        predict_X_two[0, 0, 1] = self.pre_x22

        # 任务三編碼器的输入
        for j, value in enumerate(self.x31):
            X_three[0, j, 0] = value
        for k, value in enumerate(self.x32):
            X_three[0, k, 1] = value
        for l, value in enumerate(self.code_3y):
            X_three[0, l, 2] = (value - min(self.decode_3y)) / (max(self.decode_3y) - min(self.decode_3y))
        # 解碼器輸入
        predict_X_three[0, 0, 0] = self.pre_x31
        predict_X_three[0, 0, 1] = self.pre_x32

        # 利用模型预测
        if self.flag == 1:
            predict_mean = self.pre_mean_one.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
                                                             self.x_three: X_three, self.y_one: Y_one,
                                                             self.y_two: Y_two, self.y_three: Y_three,
                                                             self.x_t_one: predict_X_one, self.x_t_two: predict_X_two,
                                                             self.x_t_three: predict_X_three})
            predict_sigma = self.pre_sigma_one.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
                                                               self.x_three: X_three, self.y_one: Y_one,
                                                               self.y_two: Y_two, self.y_three: Y_three,
                                                               self.x_t_one: predict_X_one, self.x_t_two: predict_X_two,
                                                               self.x_t_three: predict_X_three})
        elif self.flag == 2:
            predict_mean = self.pre_mean_two.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
                                                             self.x_three: X_three,
                                                             self.y_one: Y_one, self.y_two: Y_two,
                                                             self.y_three: Y_three,
                                                             self.x_t_one: predict_X_one, self.x_t_two: predict_X_two,
                                                             self.x_t_three: predict_X_three})
            predict_sigma = self.pre_sigma_two.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
                                                               self.x_three: X_three,
                                                               self.y_one: Y_one, self.y_two: Y_two,
                                                               self.y_three: Y_three,
                                                               self.x_t_one: predict_X_one, self.x_t_two: predict_X_two,
                                                               self.x_t_three: predict_X_three})
        else:
            ### ?? self.pre_mean_three ????
            #predict_mean = self.pre_mean_one.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
            predict_mean = self.pre_mean_three.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
                                                             self. x_three: X_three,
                                                             self.y_one: Y_one, self.y_two: Y_two,
                                                             self.y_three: Y_three,
                                                             self.x_t_one: predict_X_one, self.x_t_two: predict_X_two,
                                                             self.x_t_three: predict_X_three})
            #predict_sigma = self.pre_sigma_one.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
            predict_sigma = self.pre_mean_three.eval(feed_dict={self.x_one: X_one, self.x_two: X_two,
                                                               self.x_three: X_three,
                                                               self.y_one: Y_one, self.y_two: Y_two,
                                                               self.y_three: Y_three,
                                                               self.x_t_one: predict_X_one, self.x_t_two: predict_X_two,
                                                               self.x_t_three: predict_X_three})
        return predict_mean, predict_sigma

    def close_sess(self):
        tf.reset_default_graph()
        self.sess.close()

