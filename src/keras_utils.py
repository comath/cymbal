import fnmatch
import os
import re
import numpy as np
import io
import mapperWrap
from nnMapInternalUtils import get_activations,getDenseLayerDict,getNextDenseIndex
import keras.layers as KL

def getNextDenseIndex(layer,denseDict):
	'''
	Returns the index of the next dense layer, meant to help skip activation/dropout layers. 
	'''
	denceLayerIndexes = denseDict.keys()
	denceLayerIndexes = sorted(denceLayerIndexes)
	x = denceLayerIndexes.index(layer)
	if x == len(denceLayerIndexes) - 1:
		return None
	else:
		return denceLayerIndexes[x+1]

def getPenultimateLayer(denseDict):
	'''
	Returns the index of the penultimate layer.
	'''
	return denseDict.keys()[-2]

def getDenseLayerDict(model):
	"""
	Returns a dictionary of dense layers, keys are the indexes and the values are the layers themselves
	"""
	denseLayerDict = {}
	for i,modelLayer in enumerate(model.layers):
		if isinstance(modelLayer, KL.Dense):
			denseLayerDict[i] = modelLayer
	return denseLayerDict

def getActivations(model, X_batch, layer_idx, batchSize = 500):
	'''
	Takes a sequential keras model and outputs a numpy array

	Returns:
	the activations of the layer, the matrix and bias of the current layer and the selector of the next layer. 
	'''

	if layer > 0:
		_get_activations = K.function([model.layers[0].input,K.learning_phase()], [model.layers[layer_idx].output,])
		
		if X_batch.shape[0] < batchSize:
			activations = _get_activations([X_batch,0])
			return activations[0]
		else:
			activations = _get_activations([X_batch[:batchSize,],0])[0]
			for i in range(batchSize,X_batch.shape[0],batchSize):
				activations = np.concatenate([activations,_get_activations([X_batch[i:i+batchSize,],0])[0]])
			return activations
	else:
		return X_batch

def GetHyperplaneMat(model, layer):
	"""
	Returns the weights of the current layer
	"""
	denceLayers = getDenseLayerDict(model)

	hyperplaneMat, hyperplaneBias = denceLayers[layer].get_weights()
	return hyperplaneMat,hyperplaneBias

def GetSelectionMat(model, layer):
	"""
	Returns the weights of the next layer.
	"""
	denceLayers = getDenseLayerDict(model)

	nextDenseLayer = getNextDenseIndex(layer,denceLayers)
	if nextDenseLayer is None:
		raise AttributeError("No next layer")
	selectionMat, selectionBias = denceLayers[nextDenseLayer].get_weights()
	return selectionMat
