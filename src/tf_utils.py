from __future__ import print_function
import tensorflow as tf
import numpy as np

def singlePointKNN(input,pointCloud,K):
	"""
	returns a single set of the k nearest points to a query. Simple, and slow.
	"""
	distOp = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pointCloud, input)),axis=1))) 
	values,indices=tf.nn.top_k(distOp,k=K,sorted=True)
	return tf.gather(pointCloud,indices)


def fieldsOp(self,arrayOfLocal,numFields,fieldSize,stepSize, graphToMatrix,returnPoints = None):
	"""
	Returns the set of hyperlocal fields I mostly use in the work. Takes in the parameters and the tensor 	
	"""
	returnFields = []
	if returnPoints:
		retPoints = []
	arrayOfLocal = tf.unstack(arrayOfLocal)
	for localPoints in arrayOfLocal:
		# Get the local area
		if returnPoints:
			retHLPoints = []
		hyperlocalMatrices = []
		"""
		Make the numFields worth of tiny adj matrices, and save the points, if we need it.

		This cannot do fancy ordering, just straight up distance.
		"""
		for i in range(numFields):
			# Get the hyperlocal points, stepping by stepSize
			hyperlocalPoints = singlePointKNN(localPoints[i*stepSize,:],localPoints,fieldSize)
			if returnPoints:
				retHLPoints.append(hyperlocalPoints)
			# Turn the set of hyperlocal points into a distance/adj/laplacian matrix
			distMatrix = graphToMatrix(hyperlocalPoints,fieldSize)
			# Save the tiny matrix
			hyperlocalMatrices.append(distMatrix)
		if returnPoints:
			retPoints.append(tf.stack(retHLPoints,0))
		# Stack the tiny matrices
		hyperlocalMatrices = tf.stack(hyperlocalMatrices,0)
		returnFields.append(hyperlocalMatrices)
	if returnPoints:
		retPoints = tf.stack(retPoints,0)
		return tf.stack(returnFields,0), retPoints
	# Stack the stacks of tiny matrices
	return tf.stack(returnFields,0)

def matrixOp(self,arrayOfLocal, K, graphToMatrix, returnPoints = False):
	"""
	Returns the distance/adjacency/laplacian matrix for the k nearest points along with the associated points
	"""
	splitQuery = tf.unstack(queryPoints)
	results = []
	if returnPoints:
		retPoints = []

	for localPoints in arrayOfLocal:

		if returnPoints:
			retPoints.append(localPoints)
		localMatrix = graphToMatrix(localPoints,K)
		results.append(localMatrix)
	distMatrices = tf.stack(results,0)
	if returnPoints:
		retPoints = tf.stack(retPoints,0)
		return distMatrices, retPoints
	return distMatrices