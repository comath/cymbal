#ifndef _coo_graphTools_h
#define _coo_graphTools_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>

#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#endif

#ifdef USE_OPENBLAS
#include <lapacke.h>
#include <cblas.h>
#endif

#include "../mapperTree.h"
#include "../mapper.h"
#include "../nnLayerUtils.h"
#include "../key.h"

typedef struct GraphAttr
{
	pthread_spinlock_t edgeIndexSpinlock;
	int edgeIndex;

	uint numSelectors;
	uint numNodes;
	uint numHps;
	uint pathThreshold;
	float selectorSigma;
	float unitalSigma;

	int *pointTypes;
	int *pointIndexes;

	int *edgesI;
	int *edgesJ;
	float *unitalWeights;
	float *selectorWeights;
} GraphAttr;

GraphAttr * allocateGraphAttr( 
	uint numNodes, 
	uint numSelectors, 
	uint numHps,
	float unitalSigma, 
	float selectorSigma,
	uint pathThreshold);

void freeGraphAttr(GraphAttr * gts);

GraphAttr * c_getGraphAttr(
	location ** locArr,
	int numLoc,
	const double unitalSigma,
	const double selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * selectorLayer,  
	int pathThreshold,
	int numProc);

void getVertexAttr(GraphAttr *gts, int *pointTypesOut, int *pointClassesOut);
void getEdgeAttr(GraphAttr * gts,  float * unitalWeightsOut, float *selectorWeightsOut);
void getEdgesCOO(GraphAttr *gts, int *edgesIout, int *edgesJout);
void getEdges(GraphAttr *gts, int *edgesOut);
int getEdgeIndex(GraphAttr * gts);

void gaussianSimilarity(float * weights, int count, float sigma);
float gaussianSimilarity1(float weight,float sigma);

void printEdgeAttr(GraphAttr *gts);

#endif