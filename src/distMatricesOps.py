from __future__ import print_function
import tensorflow as tf
import numpy as np

def l2DistMatrix(sigma = None,matrixType = None):
	"""
	Makes a distance matrix out of a list of points.
	"""
	if matrixType is None and sigma is not None:
			matrixType = "adjacency"
	elif matrixType is not None:
		if not matrixType in ["adjacency", "laplacian"]:
			raise ValueError("Matrix type not valid, must be either 'adjacency' or 'laplacian'")
	elif matrixType is not None and sigma is None:
		raise ValueError("Please supply a sigma to produce your matrix type")
	def makeMatrix(points,K):
		idMatrix = tf.eye(K,name="correction")
		# Get the difference tensor. This is k x k x n and the i,j vector is the absolute value of the difference 
		expanded_a = tf.expand_dims(points, 1)
		expanded_b = tf.expand_dims(points, 0)
		distMatrix = tf.squared_difference(expanded_a - expanded_b)
		distMatrix = tf.reduced_sum(distMatrix,2)
		#distMatrix = tf.sqrt(distMatrix)

		if matrixType == "adjacency":
			distMatrix = tf.exp(-distMatrix/(sigma*sigma)) - idMatrix # A
		if matrixType == "laplacian":
			"""
			Computes L = 1 - D^{-1/2} A D^{-1/2}
			"""
			distMatrix = idMatrix-tf.exp(-distMatrix/(sigma*sigma))						# A
			inverse_degrees = tf.diag(1.0/tf.sqrt(tf.abs(tf.reduce_sum(distMatrix,0))))	# D^{-1/2}
			distMatrix = tf.matmul(distMatrix,inverse_degrees) 							# A D^{-1/2}
			distMatrix = tf.matmul(inverse_degrees,distMatrix) 							# D^{-1/2} A
			distMatrix = idMatrix + distMatrix											# 1 - D^{-1/2} A D^{-1/2}
		return distMatrix

	return makeMatrix