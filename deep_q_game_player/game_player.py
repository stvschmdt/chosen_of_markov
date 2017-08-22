# author: steve schmidt
# main game player for atari games or screenshot image learners


import numpy as np
import tensorflow as tf
import gym
import os
import time
from collections import deque
import numpy.random as rnd

import network
from logger import Logging

class Epsilon_Greedy(object):

    def __init__(self, params=None):
        if params:
            self.params=params
        else:
            self.eps_min = 0.05
            self.eps_max = 1.0
            self.eps_decay_steps = 50000
            self.n_outputs = 9
    # return max q_value with probability p, random with probability 1-p -- epsilon
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
            self.replay_memory = deque([], maxlen=self.replay_memory_size)

    def sample_memories(self,batch_size):
        indices = rnd.permutation(len(self.replay_memory))[:batch_size]
        # container for state, action, reward, next_state, continue {0,1}
        self.cols = [[],[],[],[], []]
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(self.cols, memory):
                col.append(value)
        self.cols = [np.array(col) for col in self.cols]
        return (self.cols[0], self.cols[1], self.cols[2].reshape(-1,1), self.cols[3], self.cols[4].reshape(-1,1))

class Parameters(object):

    def __init__(self, params=None):
        if params:
            self.params = params
        else:
            self.input_height = 88
            self.input_width = 80
            self.input_channels = 1
            self.n_steps = 10000 #100000
            self.training_start = 1000
            self.training_interval = 3
            self.save_steps = 50
            self.copy_steps = 25
            self.learning_rate = 0.01
            self.discount_rate = 0.95
            self.skip_start = 90
            self.batch_size = 50
            self.iteration = 0
            self.rewards = 0
            self.info = 0
            self.reward = 0
            self.checkpoint_path = './my_dqn.ckpt'
            self.done = True
	    self.env = gym.make('MsPacman-v0')
            self.mspacmancolor = np.array([201,164,74]).mean()
	    self.obs = self.env.reset()
            self.n_outputs = self.env.action_space.n


def driver():
    return 0


def main(FLAGS):
    logger = Logging()
    nn = network.Network() 
    settings = Parameters()
    greed = Epsilon_Greedy()
    replay = Replay()
    #X_state, actor_q_values, actor_vars, critic_..., copy_ops, copy_critic...
    X_state = tf.placeholder(tf.float32, shape=[None, settings.input_height, settings.input_width, settings.input_channels])
    #setup both an actor and a 'critic' network -- methodology to predict and play as cost function (y-y')
    actor_q_values, actor_vars = nn.q_network(X_state, name='q_networks/actor')
    critic_q_values, critic_vars = nn.q_network(X_state, name='q_networks/critic')
    copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_var in actor_vars.items()]
    copy_critic_to_actor = tf.group(*copy_ops)
    #x_action, q_value
    X_action = tf.placeholder(tf.int32, shape=[None])
    q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, settings.n_outputs), axis=1, keep_dims=True)
    #y, cost, globalstep, optimzier, training_op, init, saver
    y = tf.placeholder(tf.float32, shape=[None,1])
    cost = tf.reduce_mean(tf.square(y-q_value))
    global_step = tf.Variable(0,trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(settings.learning_rate)
    training_op = optimizer.minimize(cost, global_step=global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    logger.info('finished setup')
    #with tf.Session as sess: ********put this in driver
    with tf.Session() as sess:
        # check for saved session
        if os.path.isfile(settings.checkpoint_path):
            save.restore(sess, settings.checkpoint_path)
        else:
            init.run()
        while True:
            step = global_step.eval()
            if step >= settings.n_steps:
                break
            settings.iteration +=1
            if settings.done:
                logger.info('uh oh hit the done!')
                settings.obs = settings.env.reset()
                #manufactured 'fast forward'
                for skip in range(settings.skip_start):
                    obs, reward, settings.done, info = settings.env.step(0)
                state = network.preprocess_observation(obs, settings.mspacmancolor)
                #print(reward, settings.done, info)
            
            # vector of q values
            q_values = actor_q_values.eval(feed_dict={X_state: [state]})
            # epsilon greedy method to retrieve next action
            action = greed.e_greedy(q_values, step)
            # take this action
            obs, reward, settings.done, info = settings.env.step(action)
            next_state = network.preprocess_observation(obs, settings.mspacmancolor)
            # record these actions in replay memory
            replay.replay_memory.append((state, action, reward, next_state, 1.0-settings.done))
            state = next_state

            if settings.iteration < settings.training_start or settings.iteration % settings.training_interval != 0:
                continue

            X_state_val, X_action_val, rewards, X_next_state_val, continues = (replay.sample_memories(settings.batch_size))

            next_q_values = actor_q_values.eval(feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues*settings.discount_rate * max_next_q_values
            training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

            if step % settings.copy_steps == 0:
                #logger.info('copy to critic')
                copy_critic_to_actor.run()

            if step % settings.save_steps == 0:
                logger.info('saving settings of iteration %g' % (settings.iteration))
                logger.results('action, reward, done, info: ')
                print(action, reward, settings.done, info)
                saver.save(sess, settings.checkpoint_path)
    logger.info('saving settings of iteration %g' % (settings.iteration))
    logger.results('action, reward, done, info:')
    print(action, reward, settings.done, info)
    logger.info('program finished')




if __name__ == '__main__':
    main(1)
