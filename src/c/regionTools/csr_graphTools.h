#ifndef _csr_graphTools_h
#define _csr_graphTools_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
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
#include "../mapper.h"
#include "../nnLayerUtils.h"
#include "../key.h"

typedef struct SparseGraphMatrix {
	float *unitDiagWeights;
	float *selectorDiagWeights;
	float *selectorWeights;
	float *unitWeights;
	uint *IA;
	uint *JA;
	uint *Idiag;
	unsigned long int nnz;
	uint numNodes;
	uint numSel;

	float unitSigma;
	float selectorSigma;

	uint *pointTypes;
	uint *pointIndexes;
	uint *compressedIndexes;
} SparseGraphMatrix;

typedef struct csrStaging {
	unsigned int numHps;
	unsigned int numNodes;
	uint *rowDiagIndex;
	uint *rowCount;
	uint * rowCapacity;
	uint **rowColumns;
	float **rowData;
	float *rowDiagData;
	int pathThreshold;
	float unitSigma;

	unsigned long int nnz;
} csrStaging;

void c_getAdjacancyMatrixLocal(
	location ** locArr,
	int numLoc,
	const double unitalSigma,
	const double selectorSigma,
	const int width,
	nnLayer * hyperplaneLayer, 
	nnLayer * selectorLayer,
	kint *region,
	float *unitalWeightsOut, 
	float *selectorWeightsOut,
	int *pointTypesOut, 
	int *pointIndexesOut,
	const char outType);


csrStaging * c_getStagingCSR(
	location ** locArr,
	int numLoc,
	int numHp,
	float unitSigma,
	int pathThreshold,
	int numProc);

void populateUnitCSR(
	csrStaging *csrS,
	SparseGraphMatrix * csrU);

void populateCSR(
	location ** locArr,
	nnLayer *posSelLayer,
	float selectorSigma,
	csrStaging *csrS,
	SparseGraphMatrix * csr,
	int numProc);

void c_normalizedLaplacianCSR(
	SparseGraphMatrix * csr, 
	float *unitLap, 
	float *selectorLap, 
	int numProc);

SparseGraphMatrix * allocate_UnitCSR( 
	uint numNodes,  
	unsigned long int nnz,
	uint *IA,
	uint *JA,
	uint *Idiag,
	float *unitWeights,
	float *unitDiag);

SparseGraphMatrix * allocate_CSR( 
	uint numNodes,  
	unsigned long int nnz,
	uint *IA,
	uint *JA,
	uint *Idiag,
	float *unitWeights,
	float *unitDiag,
	float *selectorWeights,
	float *selectorDiag);

void printCSR(SparseGraphMatrix *csr);

void getSparseVertexAttr(location **loc, uint numLoc, int *pointTypesOut, int *pointIndexes);
int checkSymCSR(SparseGraphMatrix * csr, char verbose);

void free_csrStaging(csrStaging * csrS);


#endif