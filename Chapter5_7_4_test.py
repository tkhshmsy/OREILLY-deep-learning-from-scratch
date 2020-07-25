import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/DeepLearningFromScratch/')
import numpy as np
from official.dataset.mnist import load_mnist
from twolayernet2 import *

# データ読み込み
print("=== load mnist")
(x_training, t_training), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print("=== make network")
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
training_size = x_training.shape[0]
batch_size = 100
learning_rate = 0.1

training_loss_list = []
training_accuracy_list = []
test_accuracy_list = []
iter_per_epoch = max(training_size / batch_size, 1)

print("=== start batches")
for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(training_size, batch_size)
    x_batch = x_training[batch_mask]
    t_batch = t_training[batch_mask]
    # 勾配計算
    grad = network.gradient(x_batch, t_batch)
    # update
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # record
    loss = network.loss(x_batch, t_batch)
    training_loss_list.append(loss)
    # 1 epoch ごとに精度を計算
    if i % iter_per_epoch == 0:
        training_accuracy = network.accuracy(x_training, t_training)
        test_accuracy = network.accuracy(x_test, t_test)
        training_accuracy_list.append(training_accuracy)
        test_accuracy_list.append(test_accuracy)
        print("training accuracy, test accuracy | " + str(training_accuracy) + ", " + str(test_accuracy))
