# author: steve schmidt
# sample  temporal mdp (TDL) function with randomized state iterations


import numpy as np
import time
import numpy.random as rnd
#import tensorflow as tf

def q_value_iteration(Q, T, R, actions_matrix, learn_rate0, learn_rate, decay, discount_rate, n_iters, states_size, start):
    # init the Q matrix to zeros (passed in Q should account for inf
    s = start
    for state, actions in enumerate(actions_matrix):
        Q[state, actions] = 0.0
    # for number of iters- > perform iterative exploration of states/actions
    for iteration in range(n_iters):
        a = rnd.choice(actions_matrix[s])
        sp = rnd.choice(range(states_size), p=T[s,a])
        reward = R[s,a,sp]
        learn_rate = learn_rate0 / (1 + iteration * decay)
        Q[s,a] = learn_rate * Q[s,a] + (1-learn_rate) * (reward + discount_rate *np.max(Q[sp]))
        s = sp
    return Q


if __name__ == '__main__':

    nan=np.nan
    T = np.array([ [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], [[0.0, 1.0, 0.0], [nan,nan,nan],[0.0,0.0, 1.0]], [[nan,nan,nan], [0.8, 0.1, 0.1], [nan,nan,nan]] ])
    print('transition matrix:\n%s'%T)
    R = np.array([ [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [nan,nan,nan],[0.0,0.0, -50.0]], [[nan,nan,nan], [40.0, 0.0, 0.0], [nan,nan,nan]] ])
    print('rewards matrix:\n%s'%R)
    possible_actions = [[ 0,1,2], [0,2],[1]]
    Q = np.full((3,3), -np.inf)
    l_rate = 0.01
    l0_rate = 0.05
    ldecay_rate = 0.1
    d_rate = 0.95
    iters = 2000

    start_state = 0
    print('learning rate: %f\ndiscount rate: %f\niterations: %f'%(l_rate, d_rate, iters))
    ans = q_value_iteration(Q, T, R, possible_actions, l0_rate, l_rate, ldecay_rate, d_rate, iters, len(possible_actions), start_state)
    print('result Q matrix:\n %s'%( ans))
    print('resulting optimal actions:\n %s'%(np.argmax(Q,axis=1)))

