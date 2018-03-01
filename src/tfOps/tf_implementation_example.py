from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree

from tensorflow_implementation import tfknn, l2DistMatrix

np.random.seed(0)
X = np.random.random((100, 3))  # 10 points in 3 dimensions
tree = BallTree(X, leaf_size=2)              
dist, ind = tree.query(X[0:2,], k=3)
nn = X[ind,:]


feeder = tf.placeholder(tf.float32,[2,3])

"""
I want a C++ internal implementation of the following workflow. 
I want to create the local graph signals class when I define the rest of the 
graph, and I want to have several output modes, 3 are demonstrated here, that 
are predictable and one can initiate in the graph when building.

To populate the local graph with a point cloud I want to have an op that adds
points and a second one that compiles it into a fast M-tree/cover tree/KD tree.
If the compile option has not been called I want it to do a linear search on 
what it has. 

When it actually gets called I want the op to use the data stored in the point
cloud to build the local graph signals. These are the basics, one can do 
further proccessing in the rest of TF.
"""

sess = tf.Session()
testKnn = tfknn(100,3,sess)
queryOp = testKnn.knnOp(feeder,3)
# Returns a function that makes distance matrices from some points:
distMatrix = l2DistMatrix()
# Returns a function that makes distance matrices from some points, then puts it thru the gaussian kernel:
adjMatrix  = l2DistMatrix(sigma=1.0)
# Same, but also performs turns it into a normalized laplacian:
lapMatrix  = l2DistMatrix(matrixType="laplacian",sigma=1.0)

distOp = testKnn.matrixOp(feeder,3,distMatrix)
# Returns the Adjacency matrix, with the guassian similarity kernel
adjOp = testKnn.matrixOp(feeder,5,adjMatrix)
# Returns the laplacian, and the nearby points
lapOp,relatedPoints = testKnn.matrixOp(feeder,2,lapMatrix, returnPoints = True)
# Returns several hyperlocal matrices for each query point. Same options as adjMatrixOp
lapFieldsOp = testKnn.fieldsOp(feeder,2,3,5,lapMatrix, returnPoints = True)

distMatrices = []
for points in nn:
	distMatrices.append(distance_matrix(points,points))
distMatrices = np.stack(distMatrices)
print(distMatrices)

init = tf.global_variables_initializer()
sess.run(init)

"""
The fill and compile loop is currently done in numpy. Compile makes use of the session. 

This is what I want to replace with a custom C++ op as it cannot be easily done inside of TF.
"""
for i in range(10):
	addOp = testKnn.addPoints_np(X[10*i:10*(i+1),])
testKnn.compile()

print(sess.run([distOp],feed_dict={feeder:X[0:2,]}))
#print((sess.run(fields,feed_dict={feeder:X[0:1,]})))

