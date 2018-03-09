from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn.neighbors import BallTree
from scipy.spatial import distance_matrix

from graphDataExtractors import neuralGraphTensor, l2DistMatrix

class graphDataExtractors_tests(tf.test.TestCase):
	def setUp(self):
		self.dim = 10
		self.numNhbrs = 3
		# Known Numpy implementation:
		np.random.seed(0)
		self.X = np.random.random((self.numNhbrs, self.dim))  # 10 points in 3 dimensions
		self.feeder = tf.placeholder(tf.float32,[self.numNhbrs,self.dim])
		self.handmadeFeeder = tf.placeholder(tf.float32,[3,3])

		self.handmadeNbrs = np.array([
			[1,1,1],
			[0,1,1],
			[1,0,1]],
			dtype=np.float32)

		self.handMadeSelector = np.array([
			[2,4,8],
			[3,9,27]],
			dtype=np.float32)

		self.handmadeDistMatrix = np.array([
			[0,1,1],
			[1,0,2],
			[1,2,0]],
			dtype=np.float32)

		self.handmadeNeuralMatrix = np.array([
			[[0,1,1],
			[1,0,2],
			[1,2,0]],
			[
			[0,2,4],
			[4,0,6],
			[4,6,0]],
			[
			[0,3,9],
			[3,0,12],
			[9,12,0]]],
			dtype=np.float32)


	def test_l2DistMatrix_normal_handmade(self):
		"""Checks that the matrix op returns the correct shaped result"""
		with self.test_session() as sess:
			x = l2DistMatrix(self.handmadeFeeder)
			tf_nrstNbhrs = sess.run(x,feed_dict={self.handmadeFeeder:self.handmadeNbrs})
			self.assertAllClose(handmadeDistMatrix, x)

	def test_neuralGraphTensor_run(self):
		"""Checks that the fields op returns the correct shaped result"""
		np_distMatrix = distance_matrix(X[:numNhbrs,],X[:numNhbrs,])
		with self.test_session() as sess:
			x = neuralGraphTensor(self.handmadeFeeder,self.handMadeSelector)
			tf_nrstNbhrs = sess.run(x,feed_dict={self.handmadeFeeder:self.handmadeNbrs})
			self.assertAllClose(resultShape, x)

if __name__ == '__main__':
	tf.test.main()
