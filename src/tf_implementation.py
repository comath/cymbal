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

def neuralGraphWeights(nextDenceMatrix):
	"""
	Gets the full set of neural graph weights from a set of points. 
	"""
	if matrixType is None and sigma is not None:
		matrixType = "adjacency"
	elif matrixType is not None:
		if not matrixType in ["adjacency", "laplacian"]:
			raise ValueError("Matrix type not valid, must be either 'adjacency' or 'laplacian'")
	elif matrixType is not None and sigma is None:
		raise ValueError("Please supply a sigma to produce your matrix type")

	def makeMatrix(points,K):
		# Get the difference tensor. This is k x k x n and the i,j vector is the absolute value of the difference 
		expanded_a = tf.expand_dims(points, 1)
		expanded_b = tf.expand_dims(points, 0)
		diff = tf.abs(expanded_a - expanded_b)
		# Get the unit weights, and prepares it for concatenation
		unitWeights = tf.expand_dims(tf.reduce_sum(diff,2),0)
		# Gets the selection weights from the selection and the difference tensor
		selectionWeights = tf.tensordot(tf.abs(nextDenceMatrix),diff,[[0],[2]])
		#returns the concatenation.
		return tf.concat([unitWeights,selectionWeights],0)
	return makeMatrix


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


	def pointQueryOp(self,queryPoints,K):
		"""
		returns the nearest K points to each of the points in query points
		"""
		splitQuery = tf.unstack(queryPoints)
		results = []
		for x in splitQuery:
			results.append(singlePointKNN(x,self.tfPointCloud,K))
		return tf.stack(results,0)

def matrixOp(self,localPoints, K, graphToMatrix, returnPoints = False):
	"""
	Returns the distance/adjacency/laplacian matrix for the k nearest points along with the associated points
	"""
	splitQuery = tf.unstack(localPoints)
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

def fieldsOp(self,localPoints,numFields,fieldSize,stepSize, graphToMatrix,returnPoints = None):
	"""
	Returns the set of hyperlocal fields I mostly use the work
	"""
	returnFields = []
	if returnPoints:
		retPoints = []
	splitQuery = tf.unstack(localPoints)
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