import numpy as np
from hmm_forward import *


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
        out, cache_Pi, cache_prob = hmm_forward(self.A, self.B, self.Pi, self.O)
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



