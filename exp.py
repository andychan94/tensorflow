import tensorflow as tf
import numpy as np
import time
import csv
import sys

N = 1000
N_HL = 12
accuracy = 98

dim = {}
for i in range(1, N_HL + 1 + 1):
    dim[str(i)] = 0

if (N_HL > 5):
    for i in range(1, N_HL + 1 + 1):
        dim[str(i)] = 3
else:
    dim[str(1)] = 3
    dim[str(2)] = 3
    dim[str(3)] = 3
    dim[str(4)] = 3
    dim[str(5)] = 3

dim[str(0)] = 1
dim[str(N_HL + 1)] = 1

Error = 0 in dim.values()
if Error:
    sys.exit("Error!")

W_nodes = []
b_nodes = []

for iW in range(0, N_HL + 1 + 1):
    W_nodes.append(dim[str(iW)])
for ib in range(1, N_HL + 1 + 1):
    b_nodes.append(dim[str(ib)])

W_dim = {}
b_dim = {}
for i in range(1, N_HL + 1 + 1):
    W_dim[str(i)] = [dim[str(i - 1)], dim[str(i)]]
    b_dim[str(i)] = [dim[str(i)]]


def Write(accuracy):
    # W = {}
    # b = {}
    W_val = {}
    b_val = {}
    # sess = tf.Session()

    for i in range(1, N_HL + 1 + 1):
        W_val[str(i)] = sess.run(W[str(i)])
        b_val[str(i)] = sess.run(b[str(i)])
    with open('Wb.csv', 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# Weight"])
        writer.writerow([N_HL])
        writer.writerow(W_nodes)

        for i in range(1, N_HL + 1 + 1):
            writer.writerow(W_dim[str(i)])
            writer.writerows(W_val[str(i)])
        writer = csv.writer(f)
        writer.writerow(["# Bias"])
        writer.writerow([N_HL])
        writer.writerow(b_nodes)
        for i in range(1, N_HL + 1 + 1):
            writer.writerow(b_dim[str(i)])
            writer.writerow(b_val[str(i)])

x = {}
W = {}
b = {}
layer = {}

x = tf.placeholder(tf.float32, [N, dim[str(0)]])
layer = x

for i in range(1, N_HL + 1 + 1):
    W[str(i)] = tf.Variable(tf.random_uniform([dim[str(i - 1)], dim[str(i)]], minval=-1, maxval=1))
    b[str(i)] = tf.Variable(tf.random_uniform([dim[str(i)]], minval=-1, maxval=1))

for i in range(1, N_HL+1):
    layer = tf.tanh(tf.matmul(layer, W[str(i)]) + b[str(i)])

# layer = tf.matmul(layer, W[str(N_HL + 1)]) + b[str(N_HL + 1)]
y = tf.matmul(layer, W[str(N_HL + 1)]) + b[str(N_HL + 1)]
t = tf.placeholder(tf.float32, [N, dim[str(0)]])

loss = tf.reduce_sum(tf.square(y - t))

mape = tf.reduce_mean(tf.abs(y - t) / tf.abs(t)) * 100

train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

pi = np.pi
train_xi = np.linspace(-pi, pi, N)
train_x = np.zeros([N, dim[str(0)]])
for i in range(1, N + 1):
    train_x[i - 1, 0] = train_xi[i - 1]

train_t = (np.sin(train_x)**3+np.cos(train_x)**3)

start_time = time.time()
str(start_time)
print("--Learning--")
print("number of HL: %d" % N_HL)
n = 0
while True:
    n += 1
    sess.run(train_step, feed_dict={x: train_x, t: train_t})
    MAPE = 100 - sess.run(mape, feed_dict={x: train_x, t: train_t})
    if n % 1000 == 0:
        print("Step: %d, MAPE: %f" % (n, MAPE))

    if MAPE > accuracy:
        Write(accuracy)
        print("Step: %d, MAPE: %f" % (n, MAPE))
        break

end_time = time.time()
str(end_time)
print("--Finished--")
print("Time: " + str(end_time - start_time))
