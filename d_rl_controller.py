# %%

import logging
import csv
import numpy as np
import tensorflow as tf
import sys

from c_precompression_extract_joint_training import train_prune
from d_rl_input import controller_params, pruning_number_list, block_size
import termplotlib as tpl
import copy
import random
from datetime import datetime
import time
import torch
import os

from f_transformer_model import TransformerModel,PositionalEncoding
from e_compute_multiply_time_blocksize_CPU import frequency_time, pruning_number_time_dict,frequency_list, usage_time_level, normalization

from d_rl_input import prune_ratios


# logger = logging.getLogger(__name__)


def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    指数平均移动的reward
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]#卷积函数库
    return a[-1]


class Controller(object):
    def __init__(self):
        self.graph = tf.Graph()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config, graph=self.graph)

        self.model = controller_params['model']
        self.epochs = controller_params['epochs']
        self.hidden_units = controller_params['hidden_units']

        self.nn1_search_space = controller_params['sw_space']
        self.level_search_space = controller_params['level_space']

        self.nn1_num_para = len(self.nn1_search_space)
        self.level_num_para = len(self.level_search_space)

        self.num_para = self.nn1_num_para + self.level_num_para#总parameter数

        self.nn1_beg, self.nn1_end = 0, self.nn1_num_para#nn起始位置
        self.level_beg, self.level_end = self.nn1_end, self.nn1_end + self.level_num_para

        self.para_2_val = {}
        idx = 0
        for hp in self.nn1_search_space:
            self.para_2_val[idx] = hp#{idx:sw_space}
            idx += 1
        for hp in self.level_search_space:
            self.para_2_val[idx] = hp
            idx += 1
        # print("---para_2_val:",self.para_2_val)


        self.RNN_classifier = {}
        self.RNN_pred_prob = {}
        with self.graph.as_default():
            self.build_controller()

        self.reward_history = []
        self.architecture_history = []
        self.trained_network = {}

        self.explored_info = {}

    def build_controller(self):
        # logger.info('Building RNN Network')
        # Build inputs and placeholders
        with tf.name_scope('controller_inputs'):#定义一块名为controller_input的区域，并在其中工作
            # Input to the NASCell    placeholder先占位
            self.child_network_paras = tf.compat.v1.placeholder(tf.int64, [None, self.num_para], name='controller_input')
            # Discounted rewards
            self.discounted_rewards = tf.compat.v1.placeholder(tf.float32, (None,), name='discounted_rewards')
            # WW 12-18: input: the batch_size variable will be used to determine the RNN batch
            self.batch_size = tf.compat.v1.placeholder(tf.int32, [], name='batch_size')

        with tf.name_scope('embedding'):
            self.embedding_weights = []
            # share embedding weights for each type of parameters
            embedding_id = 0
            para_2_emb_id = {}
            for i in range(len(self.para_2_val.keys())):
                additional_para_size = len(self.para_2_val[i])
                additional_para_weights = tf.compat.v1.get_variable('state_embeddings_%d' % (embedding_id),
                                                          shape=[additional_para_size, self.hidden_units],
                                                          initializer=tf.initializers.random_uniform(-1., 1.))
                self.embedding_weights.append(additional_para_weights)
                para_2_emb_id[i] = embedding_id
                embedding_id += 1

            self.embedded_input_list = []
            for i in range(self.num_para):
                self.embedded_input_list.append(
                    tf.nn.embedding_lookup(self.embedding_weights[para_2_emb_id[i]], self.child_network_paras[:, i]))#选取一个张量里面索引对应的元素
            self.embedded_input = tf.stack(self.embedded_input_list, axis=-1)#矩阵拼接
            self.embedded_input = tf.transpose(self.embedded_input, perm=[0, 2, 1])#转置，并重新排列输出维度

        # logger.info('Building Controller')
        with tf.name_scope('controller'):
            with tf.compat.v1.variable_scope('RNN'):
                #用于存储LSTM单元的state_size, zero_state和output state的元组
                nas = tf.contrib.rnn.NASCell(self.hidden_units)
                tmp_state = nas.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                init_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(tmp_state[0], tmp_state[1])

                output, final_state = tf.nn.dynamic_rnn(nas, self.embedded_input, initial_state=init_state,
                                                        dtype=tf.float32)
                tmp_list = []
                for para_idx in range(self.num_para):
                    o = output[:, para_idx, :]
                    para_len = len(self.para_2_val[para_idx])#在一种para中选择一个
                    classifier = tf.layers.dense(o, units=para_len, name='classifier_%d' % (para_idx), reuse=False)#全连接层
                    self.RNN_classifier[para_idx] = classifier
                    prob_pred = tf.nn.softmax(classifier)#激活函数
                    self.RNN_pred_prob[para_idx] = prob_pred#分类概率
                    child_para = tf.argmax(prob_pred, axis=-1)#每行最大元素索引值
                    tmp_list.append(child_para)
                self.pred_val = tf.stack(tmp_list, axis=1)


        # logger.info('Building Optimization')
        # Global Optimization composes all RNNs in one, like NAS, where arch_idx = 0
        with tf.name_scope('Optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.compat.v1.train.exponential_decay(0.99, self.global_step, 50, 0.5, staircase=True)#对lr实施指数衰减
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate)#使用RMSProp算法的Optimizer

        with tf.name_scope('Loss'):
            # We seperately compute loss of each predict parameter since the dim of predicting parameters may not be same
            for para_idx in range(self.num_para):
                if para_idx == 0:#输出层分类和输入label的loss值计算
                    self.policy_gradient_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.RNN_classifier[para_idx], labels=self.child_network_paras[:, para_idx])
                else:
                    self.policy_gradient_loss = tf.add(self.policy_gradient_loss,
                                                       tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                           logits=self.RNN_classifier[para_idx],
                                                           labels=self.child_network_paras[:, para_idx]))
                # get mean of loss
            self.policy_gradient_loss /= self.num_para
            self.total_loss = self.policy_gradient_loss
            self.gradients = self.optimizer.compute_gradients(self.total_loss)#计算梯度

            # Gradients calculated using REINFORCE
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)#global reward影响gradient
                    #修正梯度

        with tf.name_scope('Train_RNN'):
            # The main training operation. This applies REINFORCE on the weights of the Controller
            self.train_operation = self.optimizer.apply_gradients(self.gradients)#应用梯度进行计算训练
            self.update_global_step = tf.compat.v1.assign(self.global_step, self.global_step + 1, name='update_global_step')#更新值

        # logger.info('Successfully built controller')

    def child_network_translate(self, child_network):#解析选择出来的参数
        dnn_out = [[None] * len(child_network[0])]
        for para_idx in range(self.num_para):
            # print("---childnetwork[0]", child_network,para_idx,self.num_para,dnn_out)
            dnn_out[0][para_idx] = (self.para_2_val[para_idx][child_network[0][para_idx]])
        return dnn_out

    def generate_child_network(self, child_network_architecture):#得到选择的child network和对应的parameter
        with self.graph.as_default():
            feed_dict = {
                self.child_network_paras: child_network_architecture,
                self.batch_size: 1
            }
            rnn_out = self.sess.run(self.RNN_pred_prob, feed_dict=feed_dict)
            predict_child = np.array([[0] * self.num_para])
            for para_idx, prob in rnn_out.items():
                predict_child[0][para_idx] = np.random.choice(range(len(self.para_2_val[para_idx])), p=prob[0])#choose child network based on probablity
            hyperparameters = self.child_network_translate(predict_child)
            return predict_child, hyperparameters


    def plot_history(self, history, ylim=(-1, 1), title="reward"):
        x = list(range(len(history)))
        y = history
        fig = tpl.figure()
        fig.plot(x, y, ylim=ylim, width=60, height=20, title=title)
        fig.show()


    def para2interface_NN(self, Para_NN1, Para_level,model,epochs,timing_constraint):#通过选择出来的子网络参数来训练剪枝网络
        #split parameters according to layer name
        idx = 0
        every_layer_dict = {}
        for name in prune_ratios:
            if prune_ratios[name] == 0.0:
                continue
            else:
                energy_level = len(pruning_number_list)
                every_layer_dict[name] = Para_NN1[idx:idx+energy_level*4]
                idx += energy_level * 4
        #choose parameters according to pruning number
        level_para = Para_level[0]
        mask_dict_set = []
        for tag in level_para:
            pruning_rate_dict = {}
            for name in every_layer_dict:
                pruning_rate_dict[name] = every_layer_dict[name][tag*4:(tag+1)*4]
            mask_dict_set.append(pruning_rate_dict)
        #compute every latency
        three_latency = []
        three_usage_times = []
        for j in range(len(level_para)):
            latency = frequency_time(level_para[j],frequency_list[j],pruning_number_time_dict)
            three_latency.append(latency)
            times = usage_time_level[j] / latency
            three_usage_times.append(times)
        norm_times = normalization(three_usage_times)
        times_reward = sum(norm_times)

        weighted_accuracy,three_sub_accuracy = train_prune(mask_dict_set,block_size,model,epochs)

        for sub_latency in three_latency:
            if sub_latency > timing_constraint:
                reward = -1
                return weighted_accuracy, times_reward,reward,mask_dict_set,level_para
        for accuracy in three_sub_accuracy:
            if accuracy < 0.4:
                reward = -1
                return weighted_accuracy,times_reward,reward,mask_dict_set,level_para
        if three_sub_accuracy[0] > three_sub_accuracy[1] > three_sub_accuracy[2]:
            reward = weighted_accuracy + times_reward
            return weighted_accuracy, times_reward, reward,mask_dict_set,level_para
        else:
            reward = 0 + times_reward
            return weighted_accuracy, times_reward,reward,mask_dict_set,level_para


    def global_train(self):
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
        step = 0
        total_rewards = 0
        child_network = np.array([[0] * self.num_para], dtype=np.int64)
        model_replica = copy.deepcopy(self.model)


        for episode in range(controller_params['max_episodes']):
            assign_model = copy.deepcopy(model_replica)
            # logger.info(
            #     '=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=>Episode {}<=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-='.format(episode))
            print('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=>Episode {}<=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-='.format(episode))
            step += 1
            episode_reward_buffer = []
            arachitecture_batch = []
            timing_constraint = int(pruning_number_time_dict[1]) + 1

            if episode % 50 == 0 and episode != 0:
                print("************Process:**********", str(float(episode) / controller_params['max_episodes'] * 100) + "%")

            for sub_child in range(controller_params["num_children_per_episode"]):
                # Generate a child network architecture
                child_network, hyperparameters = self.generate_child_network(child_network)

                DNA_NN1 = child_network[0][self.nn1_beg:self.nn1_end]
                DNA_level = child_network[0][self.level_beg:self.level_end]

                Para_NN1 = hyperparameters[0][self.nn1_beg:self.nn1_end]
                Para_level = hyperparameters[0][self.level_beg:self.level_end]

                str_NN1 = " ".join(str(x) for x in Para_NN1)
                str_NN2 = " ".join(str(x) for x in Para_level)
                str_NNs = str_NN1 + " " + str_NN2


                # logger.info("--------->NN: {}".format(str_NNs))
                # print("--------->NN: {}".format(str_NNs))

                # logger.info('=====>Step {}/{} in episode {}: HyperParameters: {} <====='.format(sub_child,
                #                                                                                 controller_params["num_children_per_episode"],
                #                                                                                 episode,
                #                                                                                 hyperparameters))


                if str_NNs in self.explored_info.keys():
                    accuracy = self.explored_info[str_NNs][0]
                    times_reward = self.explored_info[str_NNs][1]
                    reward = self.explored_info[str_NNs][2]
                    mask_dict_set = self.explored_info[str_NNs][3]
                    level_para = self.explored_info[str_NNs][4]
                else:
                    accuracy,times_reward,reward,mask_dict_set,level_para = self.para2interface_NN(Para_NN1,Para_level,self.model,self.epochs,timing_constraint)
                    self.explored_info[str_NNs] = {}
                    self.explored_info[str_NNs][0] = accuracy
                    self.explored_info[str_NNs][1] = times_reward
                    self.explored_info[str_NNs][2] = reward
                    self.explored_info[str_NNs][3] = mask_dict_set
                    self.explored_info[str_NNs][4] = level_para


                # logger.info("====================Results=======================")
                # logger.info("--------->Accuracy: {},time reward:{}".format(accuracy, times_reward))
                # logger.info("--------->Reward: {}".format(reward))
                # logger.info("=" * 90)
                torch.set_printoptions(threshold=15000)
                if episode == controller_params['max_episodes']-1:
                    print('--------->mask_dict_set:{}'.format(mask_dict_set))
                print("====================Results=======================")
                print('--------->Episode: {},energy_level:{}'.format(episode,level_para))
                print("--------->Accuracy: {},time reward:{}".format(accuracy, times_reward))
                print("--------->Reward: {}".format(reward))
                print("=" * 90)


                for name, weight in assign_model.named_parameters():
                    for original_name, original_weight in self.model.named_parameters():
                        if name == original_name:
                            original_weight.data = weight.data

                episode_reward_buffer.append(reward)#每个子network的reward
                identified_arch = np.array(
                    list(DNA_NN1) + list(DNA_level))
                arachitecture_batch.append(identified_arch)#参数列表

            current_reward = np.array(episode_reward_buffer)

            mean_reward = np.mean(current_reward)#求平均
            self.reward_history.append(mean_reward)
            self.architecture_history.append(child_network)
            total_rewards += mean_reward

            baseline = ema(self.reward_history)#reward指数平均移动
            last_reward = self.reward_history[-1]
            rewards = [last_reward - baseline]#global rewards

            feed_dict = {
                self.child_network_paras: arachitecture_batch,
                self.batch_size: len(arachitecture_batch),
                self.discounted_rewards: rewards
            }

            with self.graph.as_default():#返回一个上下文管理器，这个上下管理器使用这个图作为默认的图
                _, _, loss, lr, gs = self.sess.run(
                    [self.train_operation, self.update_global_step, self.total_loss, self.learning_rate,
                     self.global_step], feed_dict=feed_dict)

            # logger.info('=-=-=-=-=-=>Episode: {} | Loss: {} | LR: {} | Mean R: {} | Reward: {}<=-=-=-=-='.format(
            #     episode, loss, (lr, gs), mean_reward, rewards))
            print('=-=-=-=-=-=>Episode: {} | Loss: {} | LR: {} | Mean R: {} | Reward: {}<=-=-=-=-='.format(
                episode, loss, (lr, gs), mean_reward, rewards))

        print("reward history:",self.reward_history)
        # self.plot_history(self.reward_history, ylim=(min(self.reward_history)-0.01, max(self.reward_history)-0.01))


# %%

seed = 0
torch.manual_seed(seed)
random.seed(seed)
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

# print("Begin")
controller = Controller()
controller.global_train()

# %%