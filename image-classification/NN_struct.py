#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:34:11 2017

@author: Weiyu Lee
"""

import pickle
import problem_unittests as tests
import helper
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    height, width, depth = image_shape       
    x = tf.placeholder(tf.float32, [None, height, width, depth], 'x')
    
    return x

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    y = tf.placeholder(tf.float32, [None, n_classes], 'y')
    
    return y

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return keep_prob

#tf.reset_default_graph()

# Unit Test Function
#tests.test_nn_image_inputs(neural_net_image_input)
#tests.test_nn_label_inputs(neural_net_label_input)
#tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    
    _, image_height, image_width, color_channels = x_tensor.get_shape().as_list()

    weight = tf.Variable(
        tf.random_normal([conv_ksize[0], conv_ksize[1], color_channels, conv_num_outputs], mean=0, stddev=0.01))
    bias = tf.Variable(tf.zeros([conv_num_outputs]))
    
    # Apply Convolution
    conv_layer = tf.nn.conv2d(x_tensor, weight, [1, conv_strides[0], conv_strides[1], 1], padding='SAME') # mixa tuple and list????
    # Add bias
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    # Apply activation function
    conv_layer = tf.nn.relu(conv_layer)
   
    # Set the ksize (filter size) for each dimension (batch_size, height, width, depth)   
    return tf.nn.max_pool(conv_layer, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1], 'SAME')

# Unit Test Function
#tests.test_con_pool(conv2d_maxpool)

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    _, image_height, image_width, color_channels = x_tensor.get_shape().as_list()

    # -1 means "all"
    x_tensor = tf.reshape(x_tensor, [-1, image_height*image_width*color_channels])
    
    return x_tensor

# Unit Test Function
#tests.test_flatten(flatten)

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    batch_size, tensor_size = x_tensor.get_shape().as_list()
    
    weight_fc = tf.Variable(tf.random_normal([tensor_size, num_outputs], mean=0, stddev=0.01))
    bias_fc = tf.Variable(tf.zeros([num_outputs]))
    
    x_tensor = tf.add(tf.matmul(x_tensor, weight_fc), bias_fc)
    
    return tf.nn.relu(x_tensor)

# Unit Test Function
#tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    batch_size, tensor_size = x_tensor.get_shape().as_list()
    
    weight_out = tf.Variable(tf.random_normal([tensor_size, num_outputs], mean=0, stddev=0.01))
    bias_out = tf.Variable(tf.zeros([num_outputs]))
    
    x_tensor = tf.add(tf.matmul(x_tensor, weight_out), bias_out)
    
    return x_tensor

# Unit Test Function
#tests.test_output(output)

def conv_net(x, conv_ksize, conv_strides, conv_num_outputs, pool_ksize, pool_strides):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    
#    conv_ksize_1 = [3, 3]
#    conv_strides_1 = [1, 1, 1, 1]
#    conv_num_outputs_1 = 64
#    
#    conv_ksize_2 = [3, 3]
#    conv_strides_2 = [1, 1, 1, 1]
#    conv_num_outputs_2 = 128
#    
#    conv_ksize_3 = [3, 3]
#    conv_strides_3 = [1, 1, 1, 1]
#    conv_num_outputs_3 = 256
    
    #pool_ksize = [2, 2]
    #pool_strides = [1, 2, 2, 1]
    
    #num_outputs = 2000
    
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_num_layers = len(conv_num_outputs)
      
    # Assign x as the 0 layer's output (1st cnn layer's input)
    CNN_out = list()
    CNN_out.append(x)
    
    for i in range(conv_num_layers):
        if i == 0:
            CNN_out.append(conv2d_maxpool(x, conv_num_outputs[i], conv_ksize[i], conv_strides[i], pool_ksize[i], pool_strides[i]))
        else:
            CNN_out.append(conv2d_maxpool(CNN_out[i-1], conv_num_outputs[i], conv_ksize[i], conv_strides[i], pool_ksize[i], pool_strides[i]))
        #CNN_out_2 = conv2d_maxpool(CNN_out_1, conv_num_outputs_2, conv_ksize_2, conv_strides_2, pool_ksize, pool_strides)
        #CNN_out_3 = conv2d_maxpool(CNN_out_2, conv_num_outputs_3, conv_ksize_3, conv_strides_3, pool_ksize, pool_strides)

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    CNN_out_flatten = flatten(CNN_out[-1])
    
    # TODO: return output
    return CNN_out_flatten

def fully_net(x, fully_outputs):
    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fully_num_layers = len(fully_outputs)

    # Assign x as the 0 layer's output (1st cnn layer's input)
    DNN_out = list()
    DNN_out.append(x)
    
    for i in range(fully_num_layers):
        if i == 0:
            DNN_out.append(fully_conn(x, fully_outputs[i]))
        else:
            DNN_out.append(fully_conn(DNN_out[i-1], fully_outputs[i]))
        #fc_out_2 = fully_conn(fc_out_1, 1200)
        #fc_out_3 = fully_conn(fc_out_2, 325)
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    sys_output = output(DNN_out[-1], 10)
    
    # TODO: return output
    return sys_output

#def build_NN(x, y, conv_ksize, conv_strides, conv_num_outputs, pool_ksize, pool_strides, fully_outputs):
#    ##############################
#    ## Build the Neural Network ##
#    ##############################
#    
#    # Remove previous weights, bias, inputs, etc..
#    tf.reset_default_graph()
#    
#    # Inputs
#    x = neural_net_image_input((32, 32, 3))
#    y = neural_net_label_input(10)
#    keep_prob = neural_net_keep_prob_input()
#    
#    # Model
#    conv_out = conv_net(x, conv_ksize, conv_strides, conv_num_outputs, pool_ksize, pool_strides)
#    fully_out = fully_net(conv_out, fully_outputs)
#    
#    # Name logits Tensor, so that is can be loaded from disk after training
#    logits = tf.identity(fully_out, name='logits')
#    
#    # Loss and Optimizer
#    softmax_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
#    cost = tf.reduce_mean(softmax_logits)
#    optimizer = tf.train.AdamOptimizer().minimize(cost)
#    
#    # Accuracy
#    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# Unit Test Function
#tests.test_conv_net(conv_net)