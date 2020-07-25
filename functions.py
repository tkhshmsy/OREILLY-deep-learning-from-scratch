# coding: utf-8
import numpy as np

## functions for OREILLY-deep-learning-from-scratch
## by tkhshmsy@gmail.com 

def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / ( 1 + np.exp( -x ))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    # c = np.max(a)
    # exp_a = np.exp(a - c)
    # sum_exp_a = np.sum(exp_a)
    # y = exp_a / sum_exp_a
    # return y
    a = a - np.max(a, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(a) / np.sum(np.exp(a), axis=-1, keepdims=True)


# 2乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y -t) ** 2)


# バッチ対応, ラベル/one-hot両対応, 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # one-hot対応
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


# 数値微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 勾配
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
    return grad


# 勾配降下法
def gradient_descent(f, init_x, learning_rate=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= learning_rate * grad
    return x
