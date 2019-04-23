"""
Created on 2018-04-27
Project: Test
File: md_mnist.py
...
@author: Alvin(Xinyao) Sun
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_device', type=int, default=40, help='an integer for the accumulator')
parser.add_argument('--mode', default="CPU", help='an integer for the accumulator')
args = parser.parse_args()
MODE = args.mode
NUM_DEVICE = args.num_device


# load MNIST data
mnist = input_data.read_data_sets("/Users/jiangjiawei/dataset/mnist/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels
assert (len(X_train) == len(y_train))
assert (len(X_validation) == len(y_validation))
assert (len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

X_train, y_train = shuffle(X_train, y_train)

# Define training variables
EPOCHS = 150
BATCH_SIZE = 10
IS_TRAIN = True
LR = 0.003
MODEL_NAME = "net"

# Build net
x = tf.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.placeholder(tf.int32, (None))
t_phase = tf.placeholder(tf.bool)  # is_train phase
pkeep = tf.placeholder(tf.float32)  # Dropout keep value
lr = tf.placeholder(tf.float32)  # learning rate
one_hot_y = tf.one_hot(y, 10)


def inference(x, reuse=False):
    with tf.variable_scope("feature_extraction", reuse=reuse):
        conv1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.relu, name="n64k5s1", reuse=reuse)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), padding="same", strides=(2, 2))
        conv1 = tf.layers.batch_normalization(conv1, training=t_phase, reuse=reuse)

        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="n64k3s1", reuse=reuse)
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), padding="same", strides=(2, 2))
        conv2 = tf.layers.batch_normalization(conv2, training=t_phase)

        fc0 = tf.layers.flatten(conv2)
        fc1 = tf.layers.dense(inputs=fc0, units=2048, activation=tf.nn.relu, name="fc2048", reuse=reuse)
        fc1 = tf.layers.dropout(fc1, rate=pkeep)
    with tf.variable_scope("classification"):
        logits = tf.layers.dense(inputs=fc1, units=10, activation=tf.identity, name="fc10", reuse=reuse)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    # loss_operation = tf.reduce_mean(cross_entropy, name="sof_loss")
    tf.add_to_collection(name="losses", value=cross_entropy)
    return logits


def tower_loss(scope, reuse=False):
    logits = inference(x, reuse=reuse)

    losses = tf.get_collection('losses', scope=scope)
    print(losses)
    total_loss = tf.add_n(losses, name="total_loss")

    return total_loss, logits


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


global_step = 0
device = "gpu"
if MODE == "CPU":
    device = "cpu"
opt = tf.train.GradientDescentOptimizer(lr)
reuse = False
tower_grads = []
for i in range(NUM_DEVICE):
    with tf.device('/%s:%d' % (device, i)):
        with tf.name_scope('T_%d' % i) as scope:

            loss, logits = tower_loss(scope, reuse=reuse)
            m_loss = tf.reduce_mean(loss, name="sof_loss")
            reuse = True

            grads = opt.compute_gradients(loss)

            tower_grads.append(grads)
        print("Finish T%d" % i)
    tower_grads.append(grads)
grads = average_gradients(tower_grads)

# Accuracy ops
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op = opt.apply_gradients(grads)

config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
    device_count={"CPU": NUM_DEVICE},
)

sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())


# Function for evaluation with given data
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, pkeep: 1.0, t_phase: False})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


num_examples = len(X_train)
learning_rate = 0.003
train_vars = tf.trainable_variables()
num_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
for step in range(20000):
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        _, t_loss = sess.run([train_op, m_loss], feed_dict={x: batch_x, y: batch_y, pkeep: 0.75, lr: learning_rate, t_phase: True})
        global_step += 1
        if global_step % 10 == 0:
            print("Step %d, loss = %.4f" % (global_step, t_loss))

    validation_accuracy = evaluate(X_validation, y_validation)
    test_accuracy = evaluate(X_test, y_test)
    print("EPOCH {} ...".format(step + 1))
    print("Validation Accuracy = {:.3f}, Test Accuracy= {:.3f}".format(validation_accuracy, test_accuracy))