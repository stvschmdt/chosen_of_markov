# author: steve schmidt
# sample mdp functinons for use later


import numpy as np
import time
#import tensorflow as tf

def q_value_iteration(Q, T, R, actions_matrix, learn_rate, discount_rate, n_iters, states_size):
    # init the Q matrix to zeros (passed in Q should account for inf
    for state, actions in enumerate(actions_matrix):
        Q[state, actions] = 0.0
    # for number of iters- > perform iterative exploration of states/actions
    for iteration in range(n_iters):
        Q_prev =Q.copy()
        for s in range(states_size):
            for a in actions_matrix[s]:
                Q[s,a] = np.sum([T[s,a,sp]*(R[s,a,sp]+discount_rate*np.max(Q_prev[sp])) for sp in range(states_size)])

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
    d_rate = 0.95
    iters = 100
    print('learning rate: %f\ndiscount rate: %f\niterations: %f'%(l_rate, d_rate, iters))
    ans = q_value_iteration(Q, T, R, possible_actions, l_rate, d_rate, iters, len(possible_actions))
    print('result Q matrix:\n %s'%( ans))
    print('resulting optimal actions:\n %s'%(np.argmax(Q,axis=1)))

