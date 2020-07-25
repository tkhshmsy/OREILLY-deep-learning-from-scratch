import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/DeepLearningFromScratch/')
import numpy as np
from official.dataset.mnist import load_mnist
from twolayernet2 import *

# データ読み込み
(x_training, t_training), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_training[:3]
t_batch = t_training[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))