#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proj for IE510
@author: tianning
"""

############################
# Part 1

# statsmodels: include logistic regreesion model with statistical inference
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.figsize"] = (12,8)


# loss function and gradient function for logistic regreesion
def loss(x, y, w):
    n , m = x.shape
    p = 1 / (1 + np.exp(- x @ w))
    l = - np.mean(-y * np.log(p) + (1-y) * np.log(1-p))
    return l

def grad(x, y, w):
    n , m = x.shape
    p = 1 / (1 + np.exp(- x @ w))
    d = - (y - p).T @ x / n
    return d


# the distribution of training data
dim = 5   
n_sample = 2500
# n_neg = [0, 500, 400, 175, 175, 0]
# n_pos = [0, 0, 100, 325, 325, 500]

n_neg = [0, 500, 500, 400, 100, 0]
n_pos = [0, 0,    0,  100, 400, 500]

n_neg1 = np.cumsum(n_neg)
n_pos1 = np.cumsum(n_pos)
sample_ratio = np.sum(n_neg) / n_sample

machine = 5
m_sample = n_sample // machine
batch_size = 50

############
# Synthetic data
mean0 = np.zeros(dim)
mean1 = 1.5 * np.ones(dim)
sigma = 0.5 * np.ones([dim, dim])  + 0.5 * np.diag(np.ones(dim)) 

# calculate 'true' model coefficients
sigma_inv = linalg.inv(sigma)
w_true = np.zeros(dim+1)
w_true[0] =  (mean0.T @ sigma_inv @ mean0 - mean1.T @ sigma_inv @ mean1)/2 - np.log(
        sample_ratio / (1 - sample_ratio))
w_true[1:] = sigma_inv @ (mean1 - mean0)

# generate training data
np.random.seed(100)
x0 = np.random.multivariate_normal(mean0, sigma, np.sum(n_neg))
x1 = np.random.multivariate_normal(mean1, sigma, np.sum(n_pos))

x_neg = sm.add_constant(x0)
x_pos = sm.add_constant(x1)
x = np.vstack([x_neg, x_pos])
y_neg = np.full(shape=np.sum(n_neg), fill_value=0, dtype=np.int)
y_pos = np.full(shape=np.sum(n_pos), fill_value=1, dtype=np.int)
y = np.concatenate([y_neg, y_pos])

# distribute data into local machines
x_m = []
y_m = []

for i in range(machine):
    x_new = np.vstack([x_neg[n_neg1[i]:n_neg1[i+1]], x_pos[n_pos1[i]:n_pos1[i+1]]])
    x_m.append(x_new)
    
    y_new = np.concatenate([y_neg[n_neg1[i]:n_neg1[i+1]], y_pos[n_pos1[i]:n_pos1[i+1]]])
    y_m.append(y_new)
    

# genrerating testing data
np.random.seed(123)

n_test = 2000
n_test0 = int(n_test * sample_ratio)
n_test1 = int(n_test - n_test0)
x0_t = np.random.multivariate_normal(mean0, sigma, np.sum(n_test0))
x1_t = np.random.multivariate_normal(mean1, sigma, np.sum(n_test1))
y0_t = np.full(shape=n_test0, fill_value=0, dtype=np.int)
y1_t = np.full(shape=n_test1, fill_value=1, dtype=np.int)

x_t = sm.add_constant(np.vstack([x0_t, x1_t]))
y_t = np.concatenate([y0_t, y1_t])


# fit model by built in functions
logit_model = sm.Logit(y, x).fit()
logit_model.summary()


############################
# Part 2
# Optimization logistic regression by GD

np.random.seed(124)
acc0_ls = [None] * 6
acc1_ls = [None] * 6
acc_avg_ls = [None] * 6
w_opt_ls = [None] * 6
# 1) SGD
w0 = np.zeros(dim+1)
w1 = w0.copy()
alpha = 0.01
max_iter = 10000
L0 = np.zeros(max_iter)
L1 = np.zeros(max_iter)

acc0 = np.zeros(max_iter)
acc1 = np.zeros(max_iter)

for i in range(max_iter):
    ind = np.random.choice(m_sample * machine, batch_size*machine)
    d0 = grad(x, y, w1)
    L0[i] = linalg.norm(d0)
    d  = grad(x[ind, :], y[ind], w1)
    L1[i] = linalg.norm(d)
    if i > 3000:
        alpha = 0.005
    w2 = w1 - alpha * d
    w1 = w2.copy()
    p_test = 1 / (1 + np.exp(- x_t @ w1))
    acc0[i] = np.mean(p_test[:n_test0] < 0.5)
    acc1[i] = np.mean(p_test[n_test0:] >= 0.5)

w_opt_ls[0] = w1
acc0_ls[0] = acc0
acc1_ls[0] = acc1
acc_avg_ls[0] = (acc0 * n_test0 + acc1 * n_test1) / n_test

L0_SGD = L0.copy()
L1_SGD = L1.copy()

# 2) median GD
w0 = np.zeros(dim+1)
w1 = w0.copy()
alpha = 0.01
max_iter = 10000
L0 = np.zeros(max_iter)
L1 = np.zeros(max_iter)
acc0 = np.zeros(max_iter)
acc1 = np.zeros(max_iter)

for i in range(max_iter):
    
    d  = grad(x, y, w1)
    L0[i] = linalg.norm(d)
    
    # calculate stochastic gradient for each node
    d_m = np.zeros([machine, dim+1])
    for j in range(machine):
        ind = np.random.choice(m_sample, batch_size)
        d_m[j, :] = grad(x_m[j][ind, :], y_m[j][ind], w1) 
    

    # coordinate median
    d_cen = np.median(d_m, axis = 0)
    L1[i] = linalg.norm(d_cen)
    if i > 3000:
        alpha = 0.002
    w2 = w1 - alpha * d_cen
    w1 = w2.copy()
    
    # testing accuracy
    p_test = 1 / (1 + np.exp(- x_t @ w1))
    acc0[i] = np.mean(p_test[:n_test0] < 0.5)
    acc1[i] = np.mean(p_test[n_test0:] >= 0.5)
    
w_opt_ls[1] = w1
acc0_ls[1] = acc0
acc1_ls[1] = acc1
acc_avg_ls[1] = (acc0 * n_test0 + acc1 * n_test1) / n_test
L0_medianSGD = L0.copy()
L1_medianSGD = L1.copy()


# 3) noise SGD
w0 = np.zeros(dim+1)
w1 = w0.copy()
alpha = 0.01
max_iter = 10000
L0 = np.zeros(max_iter)
L1 = np.zeros(max_iter)
noise = 0.5
acc0 = np.zeros(max_iter)
acc1 = np.zeros(max_iter)

for i in range(max_iter):
    d  = grad(x, y, w1)
    L0[i] = linalg.norm(d)
    d_m = np.zeros([machine, dim+1])
    for j in range(machine):
        ind = np.random.choice(m_sample, batch_size)
        e = np.random.normal(0, noise, size = (dim+1, ))
        d_m[j, :] = grad(x_m[j][ind, :], y_m[j][ind], w1) + e

    d_cen = np.median(d_m, axis = 0)
    L1[i] = linalg.norm(d_cen)
    if i>3000:
        alpha = 0.002
    w2 = w1 - alpha * d_cen
    w1 = w2.copy()
    # testing accuracy
    p_test = 1 / (1 + np.exp(- x_t @ w1))
    acc0[i] = np.mean(p_test[:n_test0] < 0.5)
    acc1[i] = np.mean(p_test[n_test0:] >= 0.5)

w_opt_ls[2] = w1
acc_avg = (acc0 + acc1)/2
acc0_ls[2] = acc0
acc1_ls[2] = acc1
acc_avg_ls[2] = (acc0 * n_test0 + acc1 * n_test1) / n_test
L0_noisy = L0.copy()
L1_noisy = L1.copy()


# quantile-SGD
w0 = np.zeros(dim+1)
w1 = w0.copy()
alpha = 0.01
max_iter = 10000
L0 = np.zeros(max_iter)
L1 = np.zeros(max_iter)
acc0 = np.zeros(max_iter)
acc1 = np.zeros(max_iter)

for i in range(max_iter):
    d  = grad(x, y, w1)
    L0[i] = linalg.norm(d)
    # calculate stochastic gradient for each node
    d_m = np.zeros([machine, dim+1])
    for j in range(machine):
        ind = np.random.choice(m_sample, batch_size)
        d_m[j, :] = grad(x_m[j][ind, :], y_m[j][ind], w1) 
    
    # weighted average over quantiles
    d1 = np.median(d_m, axis = 0)
    d2 = np.quantile(d_m, q = 0.25, axis = 0)
    d3 = np.quantile(d_m, q = 0.75, axis = 0)
    d_q = d1/2 + d2/4 + d3/4
    L1[i] = linalg.norm(d_q)
    if i>3000:
        alpha = 0.002
    w2 = w1 - alpha * d_q
    w1 = w2.copy()
    
    # testing accuracy
    p_test = 1 / (1 + np.exp(- x_t @ w1))
    acc0[i] = np.mean(p_test[:n_test0] < 0.5)
    acc1[i] = np.mean(p_test[n_test0:] >= 0.5)

w_opt_ls[3] = w1
acc0_ls[3] = acc0
acc1_ls[3] = acc1
acc_avg_ls[3] = (acc0 * n_test0 + acc1 * n_test1) / n_test
L0_weighted = L0.copy()
L1_weighted = L1.copy()

# quantile-SGD
w0 = np.zeros(dim+1)
w1 = w0.copy()
alpha = 0.01
max_iter = 10000
L0 = np.zeros(max_iter)
L1 = np.zeros(max_iter)
acc0 = np.zeros(max_iter)
acc1 = np.zeros(max_iter)

for i in range(max_iter):
    d  = grad(x, y, w1)
    L0[i] = linalg.norm(d)
        # calculate stochastic gradient for each node
    d_m = np.zeros([machine, dim+1])
    for j in range(machine):
        ind = np.random.choice(m_sample, batch_size)
        d_m[j, :] = grad(x_m[j][ind, :], y_m[j][ind], w1) 
        
    # coordinate median
    rand = np.random.uniform()
    if rand < 0.5:
        d_q = np.median(d_m, axis = 0)
    elif rand < 0.75:
        d_q = np.quantile(d_m, q = 0.25, axis = 0)
    else:
        d_q = np.quantile(d_m, q = 0.75, axis = 0)
    
    L1[i] = linalg.norm(d_q)
    if i > 3000:
        alpha = 0.002
    w2 = w1 - alpha * d_q
    w1 = w2.copy()
    
    # testing accuracy
    p_test = 1 / (1 + np.exp(- x_t @ w1))
    acc0[i] = np.mean(p_test[:n_test0] < 0.5)
    acc1[i] = np.mean(p_test[n_test0:] >= 0.5)

w_opt_ls[4] = w1
acc0_ls[4] = acc0
acc1_ls[4] = acc1
acc_avg_ls[4] = (acc0 * n_test0 + acc1 * n_test1) / n_test
L0_random = L0.copy()
L1_random = L1.copy()

# quantile-SGD (1/3, 1/3, 1/3)
w0 = np.zeros(dim+1)
w1 = w0.copy()
alpha = 0.01
max_iter = 10000
L0 = np.zeros(max_iter)
L1 = np.zeros(max_iter)
acc0 = np.zeros(max_iter)
acc1 = np.zeros(max_iter)

for i in range(max_iter):
    d  = grad(x, y, w1)
    L0[i] = linalg.norm(d)
        # calculate stochastic gradient for each node
    d_m = np.zeros([machine, dim+1])
    for j in range(machine):
        ind = np.random.choice(m_sample, batch_size)
        d_m[j, :] = grad(x_m[j][ind, :], y_m[j][ind], w1) 
        
    # coordinate median
    rand = np.random.uniform()
    if rand < 0.333:
        d_q = np.median(d_m, axis = 0)
    elif rand < 0.667:
        d_q = np.quantile(d_m, q = 0.25, axis = 0)
    else:
        d_q = np.quantile(d_m, q = 0.75, axis = 0)
    
    L1[i] = linalg.norm(d_q)
    if i > 3000:
        alpha = 0.002
    w2 = w1 - alpha * d_q
    w1 = w2.copy()
    
    # testing accuracy
    p_test = 1 / (1 + np.exp(- x_t @ w1))
    acc0[i] = np.mean(p_test[:n_test0] < 0.5)
    acc1[i] = np.mean(p_test[n_test0:] >= 0.5)

w_opt_ls[5] = w1
acc0_ls[5] = acc0
acc1_ls[5] = acc1
acc_avg_ls[5] = (acc0 * n_test0 + acc1 * n_test1) / n_test
L0_random2 = L0.copy()
L1_random2 = L1.copy()


############################
# Part 3: ploting

# plot
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.figsize"] = (18,8)
fig, ax = plt.subplots(2, 3)
i ,j = 0, 0
ax[i, j].semilogy(L1_SGD)    
ax[i, j].semilogy(L0_SGD)
ax[i, j].legend(["used gradient", "true gradient",])
ax[i, j].set_xlabel('iterations') 
ax[i, j].set_ylabel("2-norm of gradient function")
ax[i, j].set_title("SGD")
ax[i, j].set_ylim([1e-3, 3])

i ,j = 0, 1
ax[i, j].semilogy(L1_medianSGD)    
ax[i, j].semilogy(L0_medianSGD)
ax[i, j].legend(["used gradient", "true gradient",])
ax[i, j].set_xlabel('iterations') 
ax[i, j].set_ylabel("2-norm of gradient function")
ax[i, j].set_title("medianSGD")
ax[i, j].set_ylim([1e-3, 3])

i ,j = 0, 2
ax[i, j].semilogy(L1_noisy)    
ax[i, j].semilogy(L0_noisy)
ax[i, j].legend(["used gradient", "true gradient",])
ax[i, j].set_xlabel('iterations') 
ax[i, j].set_ylabel("2-norm of gradient function")
ax[i, j].set_title("noisy medianSGD")
ax[i, j].set_ylim([1e-3, 3])

i ,j = 1, 0
ax[i, j].semilogy(L1_weighted)    
ax[i, j].semilogy(L0_weighted)
ax[i, j].legend(["used gradient", "true gradient",])
ax[i, j].set_xlabel('iterations') 
ax[i, j].set_ylabel("2-norm of gradient function")
ax[i, j].set_title("weighted (1/4, 1/2, 1/4) quantile SGD")
ax[i, j].set_ylim([1e-3, 3])

i ,j = 1, 1
ax[i, j].semilogy(L1_random)    
ax[i, j].semilogy(L0_random)
ax[i, j].legend(["used gradient", "true gradient",])
ax[i, j].set_xlabel('iterations') 
ax[i, j].set_ylabel("2-norm of gradient function")
ax[i, j].set_title("random (1/4, 1/2, 1/4) quantile SGD")
ax[i, j].set_ylim([1e-3, 3])

i ,j = 1, 2
ax[i, j].semilogy(L1_random2)    
ax[i, j].semilogy(L0_random2)
ax[i, j].legend(["used gradient", "true gradient",])
ax[i, j].set_xlabel('iterations') 
ax[i, j].set_ylabel("2-norm of gradient function")
ax[i, j].set_title("random (1/3, 1/3, 1/3) quantile SGD")
ax[i, j].set_ylim([1e-3, 3])
plt.tight_layout()
# fig.savefig('/Users/tianning/Documents/UIUC/Course/IE510/proj/plot1_unblanced.pdf')   
plt.show()


plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (18,5.5)
fig, ax = plt.subplots(1, 3)
for i in range(6):    
    ax[0].plot(acc_avg_ls[i])

for i in range(6):    
    ax[1].plot(acc0_ls[i])
    
for i in range(6):    
    ax[2].plot(acc1_ls[i])
    
for j in range(3): 
    ax[j].legend(["SGD", "medianSGD", "noisy medianSGD",
      "weighted (1/4, 1/2, 1/4) quantile-SGD",
      "random (1/4, 1/2, 1/4) quantile-SGD",
      "random (1/3, 1/3, 1/3) quantile-SGD"])
    ax[j].set_ylabel("Testing accuracy")

ax[1].set_ylim([0.3, 1])
ax[2].set_ylim([0.3, 1])

ax[0].set_title("Testing accuracy on all testing data")
ax[1].set_title("Testing accuracy on 0-label testing data")
ax[2].set_title("Testing accuracy on 1-label testing data")
# fig.savefig('/Users/tianning/Documents/UIUC/Course/IE510/proj/plot2_unblanced.pdf')   
plt.show()

p_test0 = logit_model.predict(x_t)
a01 = np.mean(p_test0[:n_test0] < 0.5) 
a02 = np.mean(p_test0[n_test0:] >= 0.5) 
print([a01, a02, (a01+a02)/2])

np.set_printoptions(precision=3)

# compare coefficients
w_compare = [None] * 8 
w_compare[0] = w_true
w_compare[1] = logit_model.params
w_compare[2:8] = w_opt_ls

w_compare = np.array(w_compare)
w_compare

# np.savetxt('/Users/tianning/Documents/UIUC/Course/IE510/proj/coef1.csv',
#         w_compare, delimiter=',')
