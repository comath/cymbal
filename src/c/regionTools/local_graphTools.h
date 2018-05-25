#ifndef _localArr_graphTools_h
#define _localArr_graphTools_h
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>

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

#include "../nnLayerUtils.h"
#include "../key.h"	
#include "dence_graphTools.h"

typedef struct regDistPair {
	float dist;
	int regIndex;
	kint * reg;
} regDistPair;

void regKNN(kint *queryKey, uint keyLen, kint * regArr, uint numLoc, uint k, regDistPair *knn);

void makeFields_reg(
	kint * queryReg,
	kint * trainingRegs,
	uint keyLen,
	uint numLoc,
	uint stepSize,
	uint numFields,
	uint fieldSize,
	uint paddingRatio,
	float unitalSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitalFieldsOut, 
	float *selectorFieldsOut);

void batchMakeFields_reg(
	kint ** queryRegs,
	uint numQuery,
	kint * trainingRegs,
	uint keyLen,
	uint numLoc,
	uint stepSize,
	uint numFields,
	uint fieldSize,
	uint paddingRatio,
	float unitSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitalFieldsOut, 
	float *selectorFieldsOut,
	int numProc);

void makeEmbeddedFields_reg(
	kint * queryReg,
	kint * regArr,
	uint keyLen,
	uint numLoc,
	uint stepSize,
	uint numFields,
	uint fieldSize,
	uint embeddingSize,
	uint paddingRatio,
	float unitSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitEmbeddingsOut, 
	float *unitEigenValsOut,
	float *selectorEmbeddingsOut,
	float *selectorEigenValsOut);

void batchMakeEmbeddedFields_reg(
	kint ** queryRegs,
	uint numQuery,
	kint * regArr,
	uint keyLen,
	uint numLoc,
	uint stepSize,
	uint numFields,
	uint fieldSize,
	uint embeddingSize,
	uint paddingRatio,
	float unitSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitEmbeddingsOut, 
	float *unitEigenValsOut,
	float *selectorEmbeddingsOut,
	float *selectorEigenValsOut,
	int numProc);


#endif