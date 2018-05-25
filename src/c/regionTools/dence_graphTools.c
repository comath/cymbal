#include "dence_graphTools.h"

// For loc_dist pairs not location arrays
void c_getAdjacancyMatrix_reg(
	kint *knn_regs,
	uint numLoc,
	const double unitSigma,
	const double selectorSigma,
	uint keyLen,
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitWeightsOut, 
	float *selectorWeightsOut)
{
	uint numSelec = 0;
	uint numHp = 0;
	if(posSelLayer){
		numHp = posSelLayer->inDim;
		numSelec = posSelLayer->outDim;
	}

	float unitWeight = 0;
	float curSeleWeight = 0;

	for(uint i = 0; i < numLoc; i++){
		for(uint j = i+1; j < numLoc; j++){
			unitWeight = numberOfDiff(knn_regs + i*keyLen,knn_regs + j*keyLen, keyLen);
			if(unitSigma > 0)
				unitWeight = exp(-unitWeight/unitSigma);
			unitWeightsOut[i*numLoc + j] = unitWeight;
			unitWeightsOut[j*numLoc + i] = unitWeight;

			if(omittedSelector > -1){
				selectorWeightsOut[omittedSelector*numLoc*numLoc + i*numLoc + j] = 0;
				selectorWeightsOut[omittedSelector*numLoc*numLoc + j*numLoc + i] = 0;	
			}

			for(int k = 0; k < numSelec; k++){ 
				if(k != omittedSelector){
					curSeleWeight = getSelectorWeight(
							knn_regs + i*keyLen,
							knn_regs + j*keyLen,
							posSelLayer->A + k*numHp,
							numHp);
					if(selectorSigma > 0)
						curSeleWeight = exp(-curSeleWeight/selectorSigma);
					selectorWeightsOut[k*numLoc*numLoc + i*numLoc + j] = curSeleWeight;
					selectorWeightsOut[k*numLoc*numLoc + j*numLoc + i] = curSeleWeight;	
				}
				if(omittedSelector > -1){
					selectorWeightsOut[omittedSelector*numLoc*numLoc + i*numLoc + j] += curSeleWeight;
					selectorWeightsOut[omittedSelector*numLoc*numLoc + j*numLoc + i] += curSeleWeight;	
				}
			}
			if(omittedSelector > -1){
				selectorWeightsOut[omittedSelector*numLoc*numLoc + i*numLoc + j] /= numSelec;
				selectorWeightsOut[omittedSelector*numLoc*numLoc + j*numLoc + i] /= numSelec;	
			}
		}
		for(int k = 0; k < numSelec; k++){ 
			selectorWeightsOut[k*numLoc*numLoc + i*numLoc + i] = 0;
		}
		unitWeightsOut[i*numLoc + i] = 0;
	}
}
void c_normalizedLaplacian(
	uint order,
	float *weights)
{
	float degree = 0;
	for(uint i = 0; i < order; i++){
		degree = 0;
		for(uint j = 0; j < order; j++){
			degree += weights[i*order + j];
		}
		if(degree != 0){
			degree = 1/sqrt(degree);
			for(uint j = 0; j < order; j++){
				weights[i*order + j] *= degree;
				weights[j*order + i] *= degree;
			}
		}
		weights[i*order + i] = 1.0;
	}
	for(uint i = 0; i < order; i++){
		for(uint j = i+1; j < order; j++){
			weights[i*order + j] = -weights[i*order + j] ;
			weights[j*order + i] = -weights[j*order + i] ;
		}
	}
}

void c_normalizedLaplacianField(
	uint degree,
	uint numSelec,
	float *unitWeights,
	float *selectorWeights)
{
	c_normalizedLaplacian(
		degree,
		unitWeights);
	for(uint i = 0; i < numSelec; i++){
		c_normalizedLaplacian(
			degree,
			selectorWeights + i*degree*degree);
	}
}

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
	int *isuppz)
{
	memcpy(scratchSpace,unitWeightsIn,order*order*sizeof(float));

	int numEigenVals;
	int rc =  LAPACKE_ssyevr(
		LAPACK_ROW_MAJOR,
		'V',
		'I',
		'U',
		order,
		scratchSpace,
		order,
		0,
		2,
		1,
		dimEmbedding,
		0,
		&numEigenVals,
		unitEigenValsOut,
		eigenVecScratch,
		dimEmbedding,
		isuppz);
	if(rc){
		printf("EigenIssue: unit matrix value %d\n",rc);
		return rc;
	}
	cblas_sgemm (
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		order,
		dimEmbedding,
		order, 
		1.0, 
		unitWeightsIn,
		order,
		eigenVecScratch, 
		dimEmbedding,
		0,
		unitEmbeddingOut,
		dimEmbedding);

	for(uint i = 0; i < numSelec; i++){
		memcpy(scratchSpace,selectorWeightsIn + i*order*order,order*order*sizeof(float));
		rc =  LAPACKE_ssyevr(
			LAPACK_ROW_MAJOR,
			'V',
			'I',
			'U',
			order,
			scratchSpace,
			order,
			0,
			2,
			1,
			dimEmbedding,
			0,
			&numEigenVals,
			selectorEigenValsOut + i*dimEmbedding,
			eigenVecScratch,
			dimEmbedding,
			isuppz);
		if(rc){
			printf("EigenIssue: selector matrix value %d\n",rc);
			printf("Parameters: embedding dim %d, order %d, numSelec: %d\n", dimEmbedding,order,numSelec);

			return rc + 2*order*i;
		}
		cblas_sgemm (
			CblasRowMajor,
			CblasNoTrans,
			CblasNoTrans,
			order,
			dimEmbedding,
			order, 
			1.0, 
			selectorWeightsIn + i*order*order,
			order,
			eigenVecScratch, 
			dimEmbedding,
			0,
			selectorEmbeddingOut + i*order*dimEmbedding,
			dimEmbedding);
	}
	return 0;
}
// For location arrays
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
	int *pointPopulationOut)
{
	uint numHp = hyperplaneLayer->outDim;
	int numSelec = posSelLayer->outDim;
	uint keyLen = calcKeyLen(numHp);

	float unitWeight = 0;
	float curSeleWeight = 0;


	kint * key_i;
	kint * key_j;
	for(uint i = 0; i < numLoc; i++){
		key_i = locArr[i]->regKey;

		pointTypesOut[i] = locArr[i]->loc.pointTypes[0];
		pointIndexesOut[i] = locArr[i]->loc.pointIndexes[0];
		pointPopulationOut[i] = locArr[i]->loc.total;

		for(uint j = i+1; j < numLoc; j++){
			key_j = locArr[j]->regKey;
			unitWeight = numberOfDiff(key_i,key_j, keyLen);
			if(unitSigma > 0)
				unitWeight = exp(-unitWeight/unitSigma);
			unitWeightsOut[i*numLoc + j] = unitWeight;
			unitWeightsOut[j*numLoc + i] = unitWeight;

			

			for(int k = 0; k < numSelec; k++){ 
				curSeleWeight = getSelectorWeight(
						key_i,
						key_j,
						posSelLayer->A + k*numHp,
						numHp);
				if(selectorSigma > 0)
					curSeleWeight = exp(-curSeleWeight/selectorSigma);
				selectorWeightsOut[k*numLoc*numLoc + i*numLoc + j] = curSeleWeight;
				selectorWeightsOut[k*numLoc*numLoc + j*numLoc + i] = curSeleWeight;	
			}
						
		}
		for(int k = 0; k < numSelec; k++){ 
			selectorWeightsOut[k*numLoc*numLoc + i*numLoc + i] = 0;
		}
		unitWeightsOut[i*numLoc + i] = 0;
	}
}

/*
// For loc_dist pairs not location arrays
void c_getAdjacancyMatrix_INTERNAL(
	locDistPair *knn,
	uint numLoc,
	const double unitSigma,
	const double selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitWeightsOut, 
	float *selectorWeightsOut,
	int *pointTypesOut, 
	int *pointIndexesOut,
	int *pointPopulationOut)
{
	uint numHp = hyperplaneLayer->outDim;
	uint numSelec = posSelLayer->outDim;
	uint keyLen = calcKeyLen(numHp);

	float unitWeight = 0;
	float curSeleWeight = 0;


	kint * key_i;
	kint * key_j;
	for(uint i = 0; i < numLoc; i++){
		key_i = knn[i].loc->regKey;

		pointTypesOut[i] = knn[i].loc->loc.pointTypes[0];
		pointIndexesOut[i] = knn[i].loc->loc.pointIndexes[0];
		pointPopulationOut[i] = knn[i].loc->loc.total;

		for(uint j = i+1; j < numLoc; j++){
			key_j = knn[j].loc->regKey;
			unitWeight = numberOfDiff(key_i,key_j, keyLen);
			if(unitSigma > 0)
				unitWeight = exp(-unitWeight/unitSigma);
			unitWeightsOut[i*numLoc + j] = unitWeight;
			unitWeightsOut[j*numLoc + i] = unitWeight;

			if(omittedSelector > -1){
				selectorWeightsOut[omittedSelector*numLoc*numLoc + i*numLoc + j] = 0;
				selectorWeightsOut[omittedSelector*numLoc*numLoc + j*numLoc + i] = 0;	
			}

			for(uint k = 0; k < numSelec; k++){ 
				if(k != omittedSelector){
					curSeleWeight = getSelectorWeight(
							key_i,
							key_j,
							posSelLayer->A + k*numHp,
							numHp);
					if(selectorSigma > 0)
						curSeleWeight = exp(-curSeleWeight/selectorSigma);
					selectorWeightsOut[k*numLoc*numLoc + i*numLoc + j] = curSeleWeight;
					selectorWeightsOut[k*numLoc*numLoc + j*numLoc + i] = curSeleWeight;	
				}
				if(omittedSelector > -1){
					selectorWeightsOut[omittedSelector*numLoc*numLoc + i*numLoc + j] += curSeleWeight;
					selectorWeightsOut[omittedSelector*numLoc*numLoc + j*numLoc + i] += curSeleWeight;	
				}
			}
			if(omittedSelector > -1){
				selectorWeightsOut[omittedSelector*numLoc*numLoc + i*numLoc + j] /= numSelec;
				selectorWeightsOut[omittedSelector*numLoc*numLoc + j*numLoc + i] /= numSelec;	
			}
		}
		for(uint k = 0; k < numSelec; k++){ 
			selectorWeightsOut[k*numLoc*numLoc + i*numLoc + i] = 0;
		}
		unitWeightsOut[i*numLoc + i] = 0;
	}
}
*/