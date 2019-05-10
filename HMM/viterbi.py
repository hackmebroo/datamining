import numpy as np
from hmm_forward import *


def viterbi_forward(A, B, Pi, O):
    N = A.shape[0]
    M = B.shape[1]
    T = O.shape[0]
    cache_Pi = []
    cache_prob = np.zeros((N, T))
    for t in range(T):
        # 当前观察值为O[t]
        if t == 0:      # 第一天
            Pi = Pi * B[:, O[t]]
            cache_Pi.append(Pi)
        else:           # 之后的每一天
            temp = np.zeros_like(Pi)
            prob_temp = np.zeros_like(A)

            for i in range(N):
                # prob = A[i, :] * (B[:, O[t]]).T
                # temp += Pi[i] * prob
                # print(B[:,i])
                prob_temp[:, i] += A[:, i] * Pi
            max_index = np.argmax(prob_temp, axis=0)    # 求出当时每个观察值最有可能从哪个状态观察到
            cache_prob[:, t] += max_index

            for j in range(N):
                temp_index = int(cache_prob[j, t])
                prob = (A[temp_index, :] * (B[:, O[t]]).T)[j]
                temp[j] = Pi[temp_index] * prob
            Pi = temp
            cache_Pi.append(Pi)
    out = np.sum(Pi)
    return out, cache_Pi, cache_prob




class Viterbi(object):
    def __init__(self, A, B, Pi, O):
        self.A = A
        self.B = B
        self.Pi = Pi
        self.O = O

    def recursion(self, cache, t, x):
        if t == 1:
            return
        else:
            print(int(cache[x][t-1]))
            x = int(cache[x][t-1])
            t -= 1
            Viterbi.recursion(self, cache, t, x)

    def step(self):
        out, cache_Pi, cache_prob = viterbi_forward(self.A, self.B, self.Pi, self.O)
        T = cache_prob.shape[1]
        x = np.argmax(cache_Pi[T - 1])
        print(x)
        Viterbi.recursion(self, cache_prob, T, x)
        # first = np.argmax(cache_Pi[0])
        # print(first)


#ppt 例子
# A = np.array([[0.4,0.6,0],[0,0.8,0.2],[0,0,1]])
# B = np.array([[0.7,0.3],[0.4,0.6],[0.8,0.2]])
# Pi = np.array([1,0,0])
# O = np.array([0,1,0,1])


# csdn例子
# A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
# B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
# Pi = np.array([0.2,0.4,0.4])
# O = np.array([0,1,0])


# 2
# A = np.array([[0.333,0.333,0.333],[0.333,0.333,0.333],[0.333,0.333,0.333]])
# B = np.array([[0.5,0.5],[0.75,0.25],[0.25,0.75]])
# Pi = np.array([0.333,0.333,0.333])
# O = np.array([0,0,0,0,1,0,1,1,1,1])


#发烧
# A = np.array([[0.7,0.3],[0.4,0.6]])
# B = np.array([[0.5,0.4,0.1],[0.1,0.3,0.6]])
# Pi = np.array([0.6,0.4])
# O = np.array([0,1,2])


# 作业例子
A = np.array([[0.8,0.1,0.1],[0.3,0.4,0.3],[0.4,0.2,0.4]])
B = np.array([[0.8,0.1,0.1],[0.2,0.5,0.3],[0.1,0.2,0.7]])
Pi = np.array([30/47,9/47,8/47])
O = np.array([0,0,1,2,1,2,1,0])


viterbi = Viterbi(A,B,Pi,O)
viterbi.step()



