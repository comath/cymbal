/* 
This provides a slow linear search KNN query on our data. 
There are two requests, one for nested queries (asking 
for the knn of a knn), and the other for vanilla queries.
*/ 

#ifndef _regionLinearKNN_h
#define _regionLinearKNN_h
#include "key.h"

typedef struct regDistPair {
	float dist;
	int regIndex;
	kint * reg;
} regDistPair;

/*
Takes in a query location and returns the K-nearest neighboors in a struct that has the index, 
a pointer to the data and the distance to the original point.
*/
void regKNN(
	kint *queryKey, 
	int keyLen, 
	kint * regArr, 
	int numLoc, 
	int k, 
	regDistPair *knn_out);

/*
Same as above, but operates on regDistPairs instead of the raw data.
*/
void localRegKNN(kint *queryKey, 
	int keyLen, 
	regDistPair *wideKNN, 
	int wideK, 
	int thinK, 
	regDistPair *knn_out);

/* 
Copies the data to a contigous piece of memory for further proccessing to a GPU or CPU 
data matrix extractor.
*/
void moveRegs(
	regDistPair *knn, 
	int numRegs, 
	int keyLen, 
	kint * regs_out);

/*
Copies the indexes to a contigous piece of memory
*/
void moveIndexes(
	regDistPair *knn, 
	int numRegs, 
	int * indexes_out);

#endif