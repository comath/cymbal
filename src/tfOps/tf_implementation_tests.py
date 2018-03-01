from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn.neighbors import BallTree
from scipy.spatial import distance_matrix

from tensorflow_implementation import tfknn, l2DistMatrix

def graphToMatrixEmptyTestFunction(points,K):
	return tf.ones([K,K])

class nearestNeighborsTest(tf.test.TestCase):
	def setUp(self):
		self.numPoints = 500
		self.dim = 10
		self.numNhbrs = 3
		self.numQuery = 5
		# Known Numpy implementation:
		np.random.seed(0)
		self.X = np.random.random((self.numPoints, self.dim))  # 10 points in 3 dimensions
		tree = BallTree(self.X, leaf_size=2)              
		dist, ind = tree.query(self.X[0:self.numQuery,], k=self.numNhbrs)
		self.nrstNbhrs = self.X[ind,:]
		self.feeder5 = tf.placeholder(tf.float32,[self.numQuery,self.dim])
		self.feeder1 = tf.placeholder(tf.float32,[1,self.dim])
		with self.test_session() as sess:
			self.testKnn = tfknn(self.X.shape[0],self.X.shape[1],sess)
			for i in range(self.X.shape[0]/100):
				addOp = self.testKnn.addPoints_np(self.X[100*i:100*(i+1),])
			self.testKnn.compile()
		self.distMatrices = []
		for points in self.nrstNbhrs:
			self.distMatrices.append(distance_matrix(points,points))
		self.distMatrices = np.stack(self.distMatrices)

	def test_nearest_queryOp(self):
		"""Checks the nearest neighbors against the BallTree from scikitlearn"""
		with self.test_session() as sess:
			x = self.testKnn.knnOp(self.feeder5,self.numNhbrs)
			tf_nrstNbhrs = sess.run(x,feed_dict={self.feeder5:self.X[0:self.numQuery,]})
			self.assertAllClose(tf_nrstNbhrs, self.nrstNbhrs)

	def test_shape_matrixOp(self):
		"""Checks that the matrix op returns the correct shaped result"""

		resultShape = np.zeros([self.numQuery,self.numNhbrs,self.numNhbrs])
		with self.test_session() as sess:
			x = self.testKnn.matrixOp(self.feeder5, self.numNhbrs, graphToMatrixEmptyTestFunction)
			self.assertShapeEqual(resultShape, x)

	def test_distances_matrixOp(self):
		"""Checks that the distance matrix produced by l2distMatrix against scipy's"""
		distMatrixFn = l2DistMatrix()
		with self.test_session() as sess:
			matOp = self.testKnn.matrixOp(self.feeder5, self.numNhbrs, distMatrixFn)
			tf_distMatrix = sess.run(matOp,feed_dict={self.feeder5:self.X[0:5,]})
			print(tf_distMatrix[0:1,])
			print(self.distMatrices[0:1,])
			self.assertAllClose(tf_distMatrix[0:1,], self.distMatrices[0:1,])

	def test_shape_fieldsOp(self):
		"""Checks that the fields op returns the correct shaped result"""
		numFields = 2
		fieldSize = 3
		stepSize = 5
		resultShape = np.zeros([self.numQuery,numFields,fieldSize,fieldSize])
		with self.test_session() as sess:
			x = self.testKnn.fieldsOp(self.feeder5,numFields,fieldSize,stepSize, graphToMatrixEmptyTestFunction)
			self.assertShapeEqual(resultShape, x)

if __name__ == '__main__':
	tf.test.main()
