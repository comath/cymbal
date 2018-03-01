from __future__ import print_function
import tensorflow as tf
import numpy as np



class tfknn(object):
	"""
	A implementation of some local graph signals in tensorflow. 

	Uses a slow, hacky way of obtaining the knn by building up a numpy array, then copying it over into TF.
	"""
	def __init__(self, numPoints, pointDim, session, dtype = tf.float32):
		super(tfknn, self).__init__()
		self.numPoints = numPoints
		self.pointDim = pointDim
		self.sess = session
		# Meant to store the point cloud
		self.tfPointCloud = tf.Variable(tf.zeros([numPoints,pointDim], dtype = dtype))
		self.tfPointCloudPlaceholder = tf.placeholder(dtype,[numPoints,pointDim])

		self.npPointCloud = np.zeros([0,pointDim],dtype= np.float32)
		self.compileOp = tf.assign(self.tfPointCloud,self.tfPointCloudPlaceholder)

	@property
	def point_cloud(self):
		return self.tfPointCloud

	def addPoints_np(self,pointsTensor):
		"""
		Adds a batch of points to the system. 
		Hack to get the system working. I would like this to be a TF op that adds to an 
		internal array that is kept track of.
		"""
		self.npPointCloud = np.concatenate([self.npPointCloud,pointsTensor])
		return None

	def compile(self):
		"""
		Moves the NP array into the tensor array and resets the numpy array to an empty array so that we can 
		refill it in the future

		This is what I want to replace with a custom C++ op as it cannot be easily done inside of TF.	
		"""
		self.sess.run(self.compileOp,feed_dict={self.tfPointCloudPlaceholder:self.npPointCloud})
		self.npPointCloud = np.zeros([0,self.pointDim],dtype= np.float32)


	def knnOp(self,queryPoints,K):
		"""
		returns the nearest K points to each of the points in query points
		"""
		splitQuery = tf.unstack(queryPoints)
		results = []
		for x in splitQuery:
			results.append(singlePointKNN(x,self.tfPointCloud,K))
		return tf.stack(results,0)

	def matrixOp(self,queryPoints, K, graphToMatrix, returnPoints = False):
		"""
		Returns the distance/adjacency/laplacian matrix for the k nearest points along with the associated points
		"""
		splitQuery = tf.unstack(queryPoints)
		results = []
		if returnPoints:
			retPoints = []

		for x in splitQuery:
			points = singlePointKNN(x,self.tfPointCloud,K)
			if returnPoints:
				retPoints.append(points)
			localMatrix = graphToMatrix(points,K)
			results.append(localMatrix)
		distMatrices = tf.stack(results,0)
		if returnPoints:
			retPoints = tf.stack(retPoints,0)
			return distMatrices, retPoints
		return distMatrices

	def fieldsOp(self,queryPoints,numFields,fieldSize,stepSize, graphToMatrix,returnPoints = None):
		"""
		Returns the set of hyperlocal fields I mostly use the work
		"""
		returnFields = []
		if returnPoints:
			retPoints = []
		splitQuery = tf.unstack(queryPoints)
		for x in splitQuery:
			# Get the local area
			localPoints = singlePointKNN(x,self.tfPointCloud,numFields*fieldSize*stepSize)
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