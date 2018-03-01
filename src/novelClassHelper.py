import fnmatch
import os
import re
import numpy as np
import io
import mapperWrap

from keras import backend as K
from keras.models import Sequential,load_model
from keras.utils import Sequence
import keras.layers as KL
import tensorflow as tf

"""
This needs to be stripped down to not deal with adversarial classes, but just novel classes.

"""

def evalWide(op,feeder,dataForProcess,batchSize):
	"""
	Meant to batch out processes that have too much data to fit in memory at once.  
	"""
	trainingDataProcessed = sess.run(op,feed_dict={tf_X:dataForProcess[0:batchSize,]})
	for i in range(batchSize,dataForProcess.shape[0],batchSize):
		nextBatch = sess.run(op,feed_dict={tf_X:dataForProcess[i:i+batchSize,]})
		np.concatenate([trainingDataProcessed,nextBatch])
	return trainingDataProcessed

class novelMetaNetworkHelper():
	def __init__(self,trainingData,trainingLabels, layer, modelMaker=None, trainingModels=None, targetLabels = None):
		
		if targetLabels is None:
			self.targetLabels = list(range(trainingDataFilter.labels.shape[1]))
		else:
			self.targetLabels = targetLabels

		if trainingModels is None and not modelMaker is None:
			self.trainingModels = {i:modelMaker() for i in self.targetLabels}
		elif not trainingModels is None:
			self.trainingModels = trainingModels
		
		self.adversarialDataMakers = []
		self.noiseMaker = None
		self.trainingDataFilter = trainingDataFilter
		self.layer = layer

	def trainTrainingModels(self,epochs,batch_size):
		for i,trainingModel in self.trainingModels.iteritems():
			normalData, normalLabels = removeClasses(trainingData,trainingLabels,[i])
			trainingModel.fit(normalData,normalLabels, epochs=epochs, batch_size=batch_size)

	def save(self,prefix):
		for i,trainingModel in self.trainingModels.iteritems():
			trainingModel.save(prefix + "trNovel-" + str(i)+".model")

	def load(self,prefix):
		self.trainingModels = {}
		for i in self.targetLabels:
			self.trainingModels[i] = load_model(prefix + "trNovel-" + str(i)+".model")
"""
	def addAdversarialMaker(self,adversarialDataMaker):
		self.adversarialDataMakers.append(adversarialDataMaker)

	def addNoiseMaker(self,noiseMaker,numNoiseRepeats = 1,novelClassNoise = False):
		self.noiseMaker = noiseMaker
		self.numNoiseRepeats = numNoiseRepeats
		self.novelClassNoise = novelClassNoise
"""
	def getFieldsFeatures(self,
		unitalSigma,
		selectorSigma,
		fieldSize,
		stepSize,
		numFields,
		sess,
		batchSize=500):
		"""
		Currently produces the neural graph weights tensor in a fields style output. This is an implementation of algorithm 4 in the pseudocode.
		"""
		fullyProcessedGraphData = None
		fullyProcessedGraphLabels = None

		with tf.Graph() as tf_graph:
			with tf.Session() as sess:
				for i,trainingModel in self.trainingModels.iteritems():
					
					hyperplaneMatrix, hyperplaneBias = GetHyperplaneMat(trainingModel, layer)
					selectionMat,selectionBias = GetSelectionWeights(trainingModel, layer)

					tf_hyperplaneMatrix = tf.Variable(hyperplaneMatrix)
					tf_hyperplaneBias = tf.Variable(hyperplaneBias)
					tf_selectionMat = tf.Variable(selectorMatrix)
					
					tf_X = tf.placeholder(tf.float32,[None,hyperplaneMatrix.shape[0]],name='UnitInput')
					# Put a step edge function on this:
					tf_evalLevel = tf.matmul(tf_X,hyperplaneMatrix) + hyperplaneBias
					sess.run([tf_selectionMat.inializer,tf_hyperplaneMatrix.inializer,tf_hyperplaneBias.inializer])

					
					# We need to get the layer below to ensure that we don't have an activation function. If it's RELU, that's fine, but 
					trainingDataPreLevel = getActivations(trainingModel, trainingData, self.layer - 1, batchSize = batchSize)
					trainingDataProcessed = evalWide(tf_evalLevel,tf_X,trainingDataPreLevel)

					"""
					All the ducks are in a row. We can start the algorithm now
					"""

					normalData, normalLabels = removeClasses(trainingDataProcessed,trainingLabels,[i])
					novelData, novelLabels = leaveClasses(trainingDataProcessed,trainingLabels,[i])

					tempBallTree = npknn(normalData.shape[0], normalData.shape[1], sess)
					tempBallTree.add(normalData)
					tempBallTree.compile(20)

					tf_graphInput = tf.placeholder(tf.float32,[None,fieldSize*stepSize*numFields,hyperplaneMatrix.shape[0]],name='UnitInput')
					gdp = {"selectorMatrix":tf_selectionMat,"averageIndex":i}

					fields = fieldsOp(tf_graphInput,numFields,fieldSize,stepSize, neuralGraphTensor,gdp)

					normalGraphData = evalWide(fields,tf_graphInput,tempBallTree.knnOp(normalData))
					novelGraphData = evalWide(fields,tf_graphInput,tempBallTree.knnOp(novelData))

					normalGraphLabels = np.zeros(normalGraphData.shape[0])
					novelGraphLabels = np.ones(novelGraphData.shape[0])

					if fullyProcessedGraphData is None:
						fullyProcessedGraphData = np.concatenate([normalGraphData,novelGraphData])
						fullyProcessedGraphLabels = np.concatenate([normalGraphLabels,novelGraphLabels])
					else:
						fullyProcessedGraphData = np.concatenate([fullyProcessedGraphData,normalGraphData,novelGraphData])
						fullyProcessedGraphLabels = np.concatenate([fullyProcessedGraphLabels,normalGraphLabels,novelGraphLabels])

		return fullyProcessedGraphData, fullyProcessedGraphLabels