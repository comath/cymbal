#ifndef _dence_graphTools_h
#define _dence_graphTools_h
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

#include "../mapperTree.h"
#include "../nnLayerUtils.h"
#include "../key.h"

void c_getAdjacancyMatrix_loc(
	location **locArr,
	uint numLoc,
	float unitSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	float *unitWeightsOut, 
	float *selectorWeightsOut,
	int *pointTypesOut, 
	int *pointIndexesOut,
	int *pointPopulationOut);

void c_getAdjacancyMatrix_reg(
	kint *knn_regs,
	uint numLoc,
	const double unitSigma,
	const double selectorSigma,
	uint keyLen, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitWeightsOut, 
	float *selectorWeightsOut);

void c_normalizedLaplacian(
	uint degree,
	float *weights);

void c_normalizedLaplacianField(
	uint degree,
	uint numSelec,
	float *unitWeights,
	float *selectorWeights);

int c_spectralEmbedding_internal(
	int order,
	uint numSelec,
	int dimEmbedding,
	float *unitWeightsIn,
	float *unitEmbeddingOut,
	float *unitEigenValsOut,
	float *selectorWeightsIn,
	float *selectorEmbeddingOut,
	float *selectorEigenValsOut,
	float *scratchSpace,
	float *eigenVecScratch,
	int *isuppz);
#endif