# author: steve schmidt
# main game player for atari games or screenshot image learners


import numpy as np
import tensorflow as tf
import gym
import time
from collections import dequeue

class Epsilon_Greedy(object):

    def __init__(self, params=None):
        if params:
            self.params=params
        else:
            self.eps_min = 0.05
            self.eps_max = 1.0
            self.eps_decay_steps = 50000
            self.n_outputs = 9

    def e_greedy(self, q_values, step):
        self.epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if rnd.rand() < self.epsilon:
            return rnd.randint(self.n_outputs)
        else:
            return np.argmax(q_values)

class Replay(object):

    def __init__(self, params=None):
        if params:
            self.params = params
        else:
            self.replay_memory_size = 10000
            self.replay_memory = deque([], maxlen=replay_memory_size)

    def sample_memories(batch_size):
        indices = rnd.permutation(len(replay_memory))[:batch_size]
        # container for state, action, reward, next_state, continue {0,1}
        self.cols = [[],[],[],[], []]
        for idx in indices:
            memory = replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        self.cols = [np.array(col) for col in cols]
        return (self.cols[0], self.cols[1], self.cols[2].reshape(-1,1), self.cols[3], self.cols[4].reshape(-1,1))

class Parameters(object):

    def __init__(self, params=None):
        if params:
            self.params = params
        else:
            self.n_steps = 100000
            self.training_start = 1000
            self.training_interval = 3
            self.save_steps = 50
            self.copy_steps = 25
            self.discount_rate = 0.95
            self.skip_start = 90
            self.batch_size = 50
            self.iteration = 0
            self.checkpoint_path = './my_dqn.ckpt'
            self.done = True


def driver():
    return 0


def main(FLAGS):

    #X_state, actor_q_values, actor_vars, critic_..., copy_ops, copy_critic...

    #x_action, q_value
    
    #y, cost, globalstep, optimzier, training_op, init, saver

    #with tf.Session as sess: ********put this in driver??
