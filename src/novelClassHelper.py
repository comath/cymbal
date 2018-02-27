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

class novelMetaNetworkHelper():
	def __init__(self,trainingDataFilter, modelMaker=None, trainingModels=None, targetLabels = None):
		
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
	def trainTrainingModels(self,epochs,batch_size):
		for i,trainingModel in self.trainingModels.iteritems():
			novelData, normalData = self.trainingDataFilter.splitDataByClass([i])
			novelLabels, normalLabels = self.trainingDataFilter.splitLabelsByClass([i])

			trainingModel.fit(normalData,normalLabels, epochs=epochs, batch_size=batch_size)

	def save(self,prefix):
		for i,trainingModel in self.trainingModels.iteritems():
			trainingModel.save(prefix + "trNovel-" + str(i)+".model")

	def load(self,prefix):
		self.trainingModels = {}
		for i in self.targetLabels:
			self.trainingModels[i] = load_model(prefix + "trNovel-" + str(i)+".model")

	def addAdversarialMaker(self,adversarialDataMaker):
		self.adversarialDataMakers.append(adversarialDataMaker)

	def addNoiseMaker(self,noiseMaker,numNoiseRepeats = 1,novelClassNoise = False):
		self.noiseMaker = noiseMaker
		self.numNoiseRepeats = numNoiseRepeats
		self.novelClassNoise = novelClassNoise

	def makeTrainingFeaturesOp(self,):	

		self.layer = layer
		self.graphMakers = {}

		for i in self.targetLabels:
			trainingModel = self.trainingModels[i]

			novelData, normalData = self.trainingDataFilter.splitDataByClass([i])
			novelIndexes, normalIndexes = self.trainingDataFilter.splitIndexesByClass([i])
			print('Adding training data to neural map and neural graph.')
			self.graphMakers[i] = nnGraph(trainingModel, points = normalData, unitalSigma = unitalSigma, layer = self.layer, selectorSigma = selectorSigma)

			numTypes = 2
			if not len(self.adversarialDataMakers) == 0:
				numTypes += 1

			print('Preprocessing the training data')
			self.graphMakers[i].setTrainingParameters(
				numTypes,
				batchSize,
				fieldSize,
				stepSize,
				numFields,
				embeddingSize,
				paddingRatio,
				omittedSelector = i,
				layer=self.layer)
			print('Adding the artificial normal data')
			self.graphMakers[i].preprocessMetaTrainingData(normalData,0,layer = self.layer)
			print('Adding the artificial novel data')
			self.graphMakers[i].preprocessMetaTrainingData(novelData,1,layer = self.layer)

			if not self.noiseMaker is None:
				for j in range(self.numNoiseRepeats):
					print('Adding the noisy artificial normal data')
					self.graphMakers[i].preprocessMetaTrainingData(self.noiseMaker(normalData),0,layer = self.layer)
					if self.novelClassNoise:
						print('Adding the noisy artificial novel data')
						self.graphMakers[i].preprocessMetaTrainingData(self.noiseMaker(novelData),1,layer = self.layer)


			for adm in  self.adversarialDataMakers:
				print('Adding the adversarial examples generated from the artificial normal data')
				self.graphMakers[i].preprocessMetaTrainingData(adm(trainingModel,normalData),2,layer = self.layer)


		self.currentGraph = 0

	def epochLen(self):
		epochLen = 0
		for i, graph in graphMakers.iteritems():
			epochLen = epochLen + graph.epochLen()

	def getFieldsBatch(self):
		batch = self.graphMakers[self.currentGraph % len(self.targetLabels)].createTrainingFieldsBatch(layer=self.layer)
		self.currentGraph += 1
		return batch

	def getEmbeddedBatch(self):
		batch = self.graphMakers[self.currentGraph % len(self.targetLabels)].createTrainingEmbeddedBatch(layer=self.layer)
		self.currentGraph += 1
		return batch

	def getEmbeddedData(self,
		data,
		fieldSize,
		embeddingSize,
		stepSize=1,
		numFields=1,
		paddingRatio=2,
		layer=None):

		unitEmbedding = np.zeros([0,numFields,fieldSize,embeddingSize],dtype=np.float32)
		selectorEmbedding = np.zeros([0,numFields,self.trainingDataFilter.dim(),fieldSize,embeddingSize],dtype=np.float32)

		for gm in self.graphMakers:
			testBatch = targetNeuralGraphMaker.graphletEmbeddedBatch(
				data,
				fieldSize,
				embeddingSize,
				stepSize = stepSize,
				numFields = numFields,
				layer=targetLayer)

			testFeedDict = {unitX:testBatch["unit"], selectorX: testBatch["selector"]}
