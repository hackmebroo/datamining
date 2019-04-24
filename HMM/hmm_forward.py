import numpy as np


def hmm_forward(A, B, Pi, O):
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
                # 前一天的状态为i
                # print(A[i,:])
                # print(B[:,O[t]])
                # print(A[i,:] * (B[:,O[t]]).T)
                prob = A[i, :] * (B[:, O[t]]).T
                temp += Pi[i] * prob
                prob_temp[:, i] += A[:, i] * Pi
            max_index = np.argmax(prob_temp, axis=0)    # 求出当时每个状态最有可能从哪个状态转换而来
            cache_prob[:, t] += max_index
            Pi = temp
            cache_Pi.append(Pi)
    out = np.sum(Pi)
    return out, cache_Pi, cache_prob

# csdn例子
# A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
# B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
# Pi = np.array([0.2,0.4,0.4])
# O = np.array([0,1,0])


# ppt例子
A = np.array([[0.4,0.6,0],[0,0.8,0.2],[0,0,1]])
B = np.array([[0.7,0.3],[0.4,0.6],[0.8,0.2]])
Pi = np.array([1,0,0])
O = np.array([0,1,0,1])


# out,_,_ = hmm_forward(A,B,Pi,O)
# print('观察序列由HMM模型产生的概率为%f'%out)