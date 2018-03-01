import numpy as np

from sklearn.neighbors import BallTree

def removeClasses(trainingData,trainingLabels,classes):
	currentLabels = np.argmax(trainingLabels,axis=1)
	trainingIndexes = np.arange(trainingData.shape[0], dtype=np.int32)
	for c in classes:
		trainingIndexes = np.extract(currentLabels != c, trainingIndexes)
		currentLabels = currentLabels[trainingIndexes]
						
	trainingLabels = np.delete(trainingLabels,classes,1)
	return trainingData[trainingIndexes], trainingLabels[trainingIndexes]

def leaveClasses(trainingData,trainingLabels,classes):
	currentIndexes = np.array([],dtype=np.int32)
	currentLabels = np.argmax(trainingLabels,axis=1)	
	trainingIndexes = np.arange(trainingData.shape[0], dtype=np.int32)

	for c in classes:
		trainingIndexes = np.extract(currentLabels == c, trainingIndexes)
		currentIndexes = np.concatenate([currentIndexes,trainingIndexes])
	
	allClasses = list(range(trainingLabels.shape[1]))
	removeClasses = list(set(allClasses) - set(classes))
	trainingLabels = np.delete(trainingLabels,removeClasses,1)
	return trainingData[currentIndexes], trainingLabels[currentIndexes]

class npknn(object):
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
		self.infoTensor = np.zeros([0],dtype= np.float32)
		self.compileOp = tf.assign(self.tfPointCloud,self.tfPointCloudPlaceholder)

	@property
	def point_cloud(self):
		return self.tfPointCloud

	def add(self,pointsTensor,infoTensor = None):
		"""
		Adds a batch of points to the system. 
		Hack to get the system working. I would like this to be a TF op that adds to an 
		internal array that is kept track of.
		"""
		self.npPointCloud = np.concatenate([self.npPointCloud,pointsTensor])
		if not infoTensor is None:
			self.npInfoTensor = np.concatenate([self.npInfoTensor,infoTensor])
		if not (infoTensor.shape[0] == 0 or infoTensor.shape[0] == self.npPointCloud):
			raise ValueError("Please be consistent and provide either no info data or the right amount")

		return None

	def compile(self,leaf_size):
		"""
		Moves the NP array into the tensor array and resets the numpy array to an empty array so that we can 
		refill it in the future

		This is what I want to replace with a custom C++ op as it cannot be easily done inside of TF.	
		"""
		self.ballTree = BallTree(elf.npPointCloud, leaf_size=leaf_size)
		self.npPointCloud = np.zeros([0,self.pointDim],dtype= np.float32)


	def knnOp(self,queryPoints,K):
		"""
		returns the nearest K points to each of the points in query points
		"""
		dist, ind = tree.query(queryPoints, k=K)
		if infoTensor.shape[0] == 0:
			return self.npPointCloud[ind,:]
		else:
			return self.npPointCloud[ind,:],self.infoTensor[ind]
