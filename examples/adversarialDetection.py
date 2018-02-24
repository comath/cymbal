import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential,load_model
import keras.layers as KL
from nnMap.DataInjestors import mnist

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
mnist.test.images, mnist.test.labels

def denseLayer(X,dimension,name=None, activation=tf.relu):
	if activation is None:
		def emptyActivation(y):
			return y
		activation = emptyActivation
	inDim = X.shape[1]
	with tf.name_scope(name):
		weights1 = tf.Variable(tf.random_uniform([inDim,dimension], -0.005, 0.005),name='Weights')
		bias1 = tf.Variable(tf.random_uniform([dimension], -0.005, 0.005),name='Bias')
		eval1 = emptyActivation(tf.matmul(X,weights1) + bias1,name='Evaluation')
		

class leNetModel:
	def __init__(self,X):
		self.modelName = "convLeCun"

		model.add(KL.Reshape((28,28,1),input_shape=(28*28,)))
		conv1 = tf.relu(tf.conv2d(X,20,(5, 5)))
		pool1 = tf.max_pooling2d(conv1,(2,2),(2,2),name = "max_pool_1")
		conv2 = tf.relu(tf.conv2d(pool1,50,(5, 5)))
		pool2 = tf.max_pooling2d(conv2,(2,2),(2,2),name = "max_pool_2")
		self.flatten = tf.layers.flatten(pool2)
		#model.add(KL.Dropout(0.5))
		self.dense = denseLayer(flatten,500,"dense_500")
		#model.add(KL.Dropout(0.5))
		self.output = denseLayer(flatten,10,"output",tf.softmax)

	def loss(self,Y):
		return = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=self.output)


	def trainingOp(self,Y):
		optimizer = tf.train.RMSPropOptimizer(0.0001,
		    decay=0.98,
		    momentum=0.001,
		    centered=True,
		    name='RMSProp')
		globalStep = tf.Variable(0, name='globalStep', trainable=False)
		self.trainOp = optimizer.minimize(self.loss(Y), global_step=globalStep)
		return self.trainOp

