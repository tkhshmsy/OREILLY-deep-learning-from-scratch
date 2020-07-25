#!/usr/bin/python
# import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/DeepLearningFromScratch')
import numpy as np
from functions import softmax, cross_entropy_error
from functions import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


def f(W):
    return net.loss(x, t)


print("net.W")
net = SimpleNet()
print(net.W)

print("net.predict(x)")
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

np.argmax(p)
t = np.array([0, 0, 1])
net.loss(x, t)

print("numerical_gradient")
dW = numerical_gradient(f, net.W)
print(dW)
