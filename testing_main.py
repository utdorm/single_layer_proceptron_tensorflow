from os import path
from PIL import Image

import glob
import numpy as np
# import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


addition_samples = './train/add/*.png'
a_images = glob.glob(addition_samples)


addition_test = './test/add/*.png'
a_test = glob.glob(addition_test)

division_samples = './train/div/*.png'
d_images = glob.glob(division_samples)

division_test = './test/div/*.png'
d_test = glob.glob(division_test)

multiplication_samples = './train/mlt/*.png'
m_images = glob.glob(multiplication_samples)

multiplication_test = './test/mlt/*.png'
m_test = glob.glob(multiplication_test)

subtraction_samples = './train/sbt/*.png'
s_images = glob.glob(subtraction_samples)

subtraction_test = './test/sbt/*.png'
s_test = glob.glob(subtraction_test)








"""

# Parameters
learning_rate = 0.1
num_steps = 5000
display_step = 50
batch_size = 128

# Network Parameters
perceptron_Layer_1 = 1 # 1st layer number of neurons
# perceptron_Layer_1 = 100 # 1st layer number of neurons
# perceptron_Layer_1 = 25000 # 1st layer number of neurons


num_input = 784 # MNIST data input (img shape: 28*28)
# num_input = 54656 # collections data input (img shape: 224*224)

num_classes = 10 # MNIST total classes (0-9 digits)
# num_classes = 4 # collection data total classes (+ | - | / | * symbols)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'layer1': tf.Variable(tf.random_normal([num_input, perceptron_Layer_1])),
    'outlayer': tf.Variable(tf.random_normal([perceptron_Layer_1, num_classes]))
}
biases = {
    'bias1': tf.Variable(tf.random_normal([perceptron_Layer_1])),
    'outlayer': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['layer1']), biases['bias1'])
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['outlayer']) + biases['outlayer']
    return out_layer


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                "{:.4f}".format(loss) + ", Training Accuracy= " + \
                "{:.3f}".format(acc))

    

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                    Y: mnist.test.labels}))

"""