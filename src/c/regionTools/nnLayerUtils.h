#ifndef _nnLayerUtils_h
#define _nnLayerUtils_h

#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <xmmintrin.h>
#include <immintrin.h>  /* AVX2 */

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

#include "key.h"
/*
Struct for passing layers of a perceptron.
*/
typedef struct nnLayer {
	uint outDim;
	uint inDim;
	float *A;
	float *b;
} nnLayer;

// Tools for dealing with a layer
nnLayer *allocateLayer(uint inDim, uint outDim);
nnLayer * createCopyPosLayer(float * A, float *b, uint inDim, uint outDim);
nnLayer *createCopyLayer(float *A, float *b, uint outDim, uint inDim);
nnLayer *createLayer(float *A, float *b, uint outDim, uint inDim);
void freeLayer(nnLayer * layer);

// Computes Ax+b and puts the result in the output
void evalLayer(nnLayer * layer, float *x, float *output);
void printFloatArr(float * arr, uint length);
void printMatrix(float * arr, uint inDim, uint outDim);
void printLayer(nnLayer * layer);

void makeLayerPositive(nnLayer * layer1);
nnLayer * copyLayerPositive(nnLayer * layer1,char transpose);

void getSelectorWeights(kint *x, kint *y, nnLayer *selectorLayer, float *weights);
void getSelectorWeightsT(kint *x,kint *y,nnLayer *positiveSelectorLayer,float *weights, uint offset);
float getSelectorWeight(kint *x, kint *y,float * selectionVec, int dataLen);

#endif