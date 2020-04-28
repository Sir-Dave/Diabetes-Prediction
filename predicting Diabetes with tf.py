import math
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def random_mini_batches(X, Y, mini_batch_size=16, seed=0):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

data = pd.read_csv("pima_indians_diabetes.csv")

X = data[['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
          'Insulin', 'BMI', 'Pedigree', 'Age']].values.transpose()

Y = data['Class'].values
Y = Y.reshape(Y.shape[0], 1).transpose()

##Split the data into training and test set

train_X = X[:, :614]
test_X = X[:, 614:]

train_Y = Y[:, :614]
test_Y = Y[:, 614:]


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")

    return X, Y


def initialize_parameters(y_weight, x_weight):
    #W = tf.Variable(tf.random_normal((y_weight, x_weight), dtype=tf.float32))
    #b = tf.Variable(tf.zeros(y_weight))
    W = tf.get_variable("W",shape=[y_weight, x_weight], initializer=tf.glorot_uniform_initializer, dtype=tf.float32)
    b = tf.get_variable("b", shape=[y_weight,1], initializer=tf.initializers.zeros)

    parameters = {"W": W, "b": b}

    return parameters


def forward_prapagation(X, parameters):
       W = parameters["W"]
       b = parameters["b"]
       Z = tf.add(tf.matmul(W, X), b)
       #Z = tf.sigmoid(Z)

       return Z


def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=Y))

    return cost


num_epochs = 100
learning_rate = 0.001
batch_size = 2

with tf.Session() as sess:
    n_x, m = train_X.shape
    n_y = train_Y.shape[0]

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_y, n_x)
    Z = forward_prapagation(X, parameters)
    cost = compute_cost(Z, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    sess.run(init)

    epoch = 0
    num_batches = int(m / batch_size)
    for i in range(num_epochs):
        minibatches = random_mini_batches(train_X, train_Y, mini_batch_size=batch_size)

        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch

            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

            epoch += minibatch_cost / num_batches

        if i % 10 == 0:
            print("Cost after epoch %i: %f" % (i, epoch))

    parameters = sess.run(parameters)
    print("Parameters have been trained")

    sigmoid = tf.sigmoid(Z)
    predicted = tf.cast(sigmoid> 0.5, "float")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), "float"))

    print("Train Accuracy:", accuracy.eval({X: train_X, Y: train_Y}))
    print("Test Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))

    #return parameters