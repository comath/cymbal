#include "local_graphTools.h"

#define LEFTCHILD(X)  2*(X)+1
#define PARENT(X)  ((X) - 1)/2

int distOrderReg(const void * a, const void * b)
{
	const regDistPair *myA = a;
	const regDistPair *myB = b;
	
	if(myA->dist - myB->dist > 0)
		return 1;
	
	if(myA->dist - myB->dist < 0)
		return -1;

	return 0;
}

void swapPairs(uint i, uint j, regDistPair *knn)
{
	int dist = knn[i].dist;
	int regIndex = knn[i].regIndex;
	kint * reg = knn[i].reg;

	knn[i].dist = knn[j].dist;
	knn[i].regIndex = knn[j].regIndex;
	knn[i].reg = knn[j].reg;

	knn[j].dist = dist;
	knn[j].regIndex = regIndex;
	knn[j].reg = reg;
}

void siftDown(uint start, uint end, regDistPair *knn)
{
	uint root = start;
	uint swap = root;
	uint child = root;
	while(LEFTCHILD(root) < end){
		child = LEFTCHILD(root);
		swap = root;
		if(knn[swap].dist < knn[child].dist)
			swap = child;
		if(child + 1 < end && knn[swap].dist < knn[child + 1].dist)
			swap = child + 1;
		if(swap == root)
			return;
		else {
			swapPairs(swap,root,knn);
			root = swap;
		}

	}
}

void siftUp(uint start, uint end, regDistPair *knn)
{
	uint child = end;
	uint parent = child;
	while(child > start){
		parent = PARENT(child);
		if(knn[parent].dist < knn[child].dist){
			swapPairs(parent,child,knn);
			child = parent;
		} else {
			return;
		}
	}
}


void regKNN(kint *queryKey, uint keyLen, kint * regArr, uint numLoc, uint k, regDistPair *knn)
{
	int curDist = -1;
	// Build the initial heap
	for(uint i = 0; i < k; ++i){
		knn[i].dist = numberOfDiff(queryKey,regArr + i*keyLen, keyLen);
		knn[i].regIndex = i;
		knn[i].reg = regArr + i*keyLen;
		siftUp(0,i,knn);
	}

	// We have a max heap
	for(uint i = k; i < numLoc; ++i){
		curDist = numberOfDiff(queryKey,regArr + i*keyLen, keyLen);
		if(curDist < knn[0].dist){
			knn[0].dist = curDist;
			knn[0].regIndex = i;
			knn[0].reg = regArr + i*keyLen;
			siftDown(0, k - 1, knn);
		}
	}

	float * bigAdgeMat = malloc(k*k*sizeof(float));

	kint * regions = malloc(k*keyLen*sizeof(kint));
	cpRegPair(knn, regions, k, keyLen);

	c_getAdjacancyMatrix_reg(
		regions,
		k,
		2,
		1,
		keyLen, 
		NULL,
		-1,			
		bigAdgeMat, 
		NULL);	
	for(uint i = 0; i < k; ++i){
		for(uint j = 0; j < k; ++j){
			knn[i].dist += bigAdgeMat[i*k + j];
		}
	}
	qsort(knn, k, sizeof(regDistPair), distOrderReg);

	free(bigAdgeMat);
	free(regions);
}

void localRegKNN(kint *queryKey, uint keyLen, regDistPair *wideKNN, uint wideK, uint thinK, regDistPair *knn)
{
	int curDist = -1;
	// Build the initial heap
	for(uint i = 0; i < thinK; ++i){
		knn[i].dist = numberOfDiff(queryKey,wideKNN[i].reg, keyLen);
		knn[i].regIndex = wideKNN[i].regIndex;
		knn[i].reg = wideKNN[i].reg;
		siftUp(0,i,knn);

	}
	// We have a max heap
	for(uint i = thinK; i < wideK; ++i){
		curDist = numberOfDiff(queryKey,wideKNN[i].reg, keyLen);
		
		if(curDist < knn[0].dist){
			knn[0].dist = curDist;
			knn[0].regIndex = wideKNN[i].regIndex;
			knn[0].reg = wideKNN[i].reg;
			siftDown(0, thinK - 1, knn);
		}
	}

	float * bigAdgeMat = malloc(thinK*thinK*sizeof(float));

	kint * regions = malloc(thinK*keyLen*sizeof(kint));
	cpRegPair(knn, regions, thinK, keyLen);

	c_getAdjacancyMatrix_reg(
		regions,
		thinK,
		2,
		1,
		keyLen, 
		NULL,
		-1,			
		bigAdgeMat, 
		NULL);	
	for(uint i = 0; i < thinK; ++i){
		for(uint j = 0; j < thinK; ++j){
			knn[i].dist += bigAdgeMat[i*thinK + j];
		}
	}


	qsort(knn, thinK, sizeof(regDistPair), distOrderReg);
	free(bigAdgeMat);
	free(regions);

}

/*
int getFurthestIndex_reg(regDistPair *knn, uint k)
{
	int maxDistIndex = 0;
	int maxDist = knn[0].dist; 
	for(uint i = 0; i < k; ++i){
		if(knn[i].dist > maxDist){
			maxDist = knn[i].dist;
			maxDistIndex = i;
		}
	}
	return maxDistIndex;
}

// Replaces the furthest and returns the new furthest index
int replaceFurthest_reg(regDistPair *knn, uint k, int replaceIndex, kint *newLoc, int newDist, int newIndex)
{
	knn[replaceIndex].reg = newLoc;
	knn[replaceIndex].regIndex = newIndex;
	knn[replaceIndex].dist = newDist;
	return getFurthestIndex_reg(knn,k);
}

int distOrderReg(const void * a, const void * b)
{
	const regDistPair *myA = a;
	const regDistPair *myB = b;
	
	return myA->dist - myB->dist;
}

void regKNN(kint *queryKey, uint keyLen, kint * locArr, uint numLoc, uint k, regDistPair *knn, nnLayer * orderLayer)
{
	int maxDist = 0, maxDistIndex = 0;
	int curDist = -1;

	for(uint i = 0; i < k; ++i){
		knn[i].reg = locArr + i*keyLen;
		knn[i].dist = numberOfDiff(queryKey,locArr + i*keyLen, keyLen);
		knn[i].regIndex = i;
	}
	maxDistIndex = getFurthestIndex_reg(knn, k);
	maxDist = knn[maxDistIndex].dist;
	for(uint i = k; i < numLoc; ++i){

		curDist = numberOfDiff(queryKey,locArr + i*keyLen, keyLen);
		if(curDist < maxDist){

			maxDistIndex = replaceFurthest_reg(knn, k, maxDistIndex, locArr + i*keyLen, curDist, i);
			maxDist = curDist;
		}
	}
	if(orderLayer){
		float * bigAdgeMat = malloc(k*k*sizeof(float));
		float * orderinf = malloc(k*k*sizeof(float));

		kint * regions = malloc(k**keyLen*sizeof(kint));

		cpRegPair(knn, regions, k, keyLen);
		c_getAdjacancyMatrix_reg(
			regions,
			k,
			10,
			1,
			orderLayer, 
			NULL,
			-1,			// This selector index will be an average of the other selectors.
			bigAdgeMat, 
			NULL);
		for 
		evalLayer(orderLayer, , float *output);

	}


	qsort(knn, k, sizeof(regDistPair), distOrderReg);
}

void localRegKNN(kint *queryKey, uint keyLen, regDistPair *wideKNN, uint numLoc, uint k, regDistPair *knn)
{
	int maxDist = -1, maxDistIndex = 0;
	int curDist = -1;
	for(uint i = 0; i < k; ++i){
		knn[i].reg = wideKNN[i].reg;
		knn[i].dist = numberOfDiff(queryKey,wideKNN[i].reg, keyLen);
		knn[i].regIndex = wideKNN[i].regIndex;
	}
	for(uint i = k; i < numLoc; ++i){
		curDist = numberOfDiff(queryKey,wideKNN[i].reg, keyLen);
		if(curDist > maxDist){
			maxDistIndex = replaceFurthest_reg(knn, k, maxDistIndex, wideKNN[i].reg, curDist,wideKNN[i].regIndex);
			maxDist = curDist;
		}
	}
	
	qsort(knn, k, sizeof(regDistPair), distOrderReg);
}
*/
void  cpRegPair(regDistPair *knn, kint * knn_regWorkSpace, uint numRegs, uint keyLen)
{
	for(uint i = 0; i < numRegs; ++i){		
		memcpy(knn_regWorkSpace + i*keyLen, knn[i].reg,keyLen*sizeof(kint));
	}
}

void makeFields_reg_INTERNAL(
	kint * queryReg,
	kint * regArr,
	uint keyLen,
	uint numLoc,
	uint numFields,
	uint stepSize,
	uint fieldSize,
	uint paddingRatio,
	float unitSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitFieldsOut, 
	float *selectorFieldsOut,
	regDistPair *wideknn,
	regDistPair *narrowknn,
	kint * knn_regWorkSpace)
{
	uint numSelec = posSelLayer->outDim;

	wideknn[0].regIndex = -1;
	wideknn[0].dist = 0;
	wideknn[0].reg = queryReg;
	memcpy(knn_regWorkSpace, queryReg,keyLen*sizeof(kint));
	regKNN(queryReg, keyLen, regArr, numLoc, stepSize*fieldSize*paddingRatio, wideknn + 1);

	/* 	Since we use the training data to train the metanetwork we need to ensure that the training locations are 
		identical to the adversarial ones in all but weights. We do this by ensuring that if the location is already 
		saved in the graph we do not include it twice. All the training locations would have this flaw, but for the 
		following conditional. 
	*/
	if(numberOfDiff(queryReg,wideknn[1].reg, keyLen) == 0){
		#ifdef DEBUG
			printf("Match, shift copy\n");
		#endif
		wideknn = wideknn + 1;
	} else {
		#ifdef DEBUG
			printf("No Match, normal copy\n");
		#endif
	}
	cpRegPair(wideknn + 1, knn_regWorkSpace + 1, fieldSize-1, keyLen);
	#ifdef DEBUG
		printf("======\n");
		for (uint i = 0; i < stepSize*fieldSize*paddingRatio; ++i)
		{	
			printf("[%d] Dist: %f, index: %d   \t",i,wideknn[i].dist,wideknn[i].regIndex);
			printKey(wideknn[i].reg,32);
			if(i < fieldSize){
				printf("Copied Key:");
				printKey(knn_regWorkSpace + i*keyLen,32);
			}
		}
		printf("======\n");
	#endif
	c_getAdjacancyMatrix_reg(
		knn_regWorkSpace,
		fieldSize,
		unitSigma,
		selectorSigma,
		keyLen, 
		posSelLayer,
		omittedSelector,			// This selector index will be an average of the other selectors.
		unitFieldsOut, 
		selectorFieldsOut);

	for(uint i = 1; i < numFields; ++i){
		localRegKNN(regArr + keyLen*wideknn[i*stepSize].regIndex, keyLen, wideknn, stepSize*fieldSize*paddingRatio, fieldSize, narrowknn);
		cpRegPair(narrowknn, knn_regWorkSpace, fieldSize, keyLen);
		#ifdef DEBUG
			printf("==========\n");
			for (uint j = 0; j < fieldSize; ++j)
			{	
				printf("[%d] Dist: %f, index: %d   \t",j,narrowknn[j].dist,narrowknn[j].regIndex);
				printKey(narrowknn[j].reg,32*keyLen);
				printf("Copied Key:");
				printKey(knn_regWorkSpace + j*keyLen,32*keyLen);
			}
			printf("======\n");

		#endif
		c_getAdjacancyMatrix_reg(
			knn_regWorkSpace,
			fieldSize,
			unitSigma,
			selectorSigma,
			keyLen, 
			posSelLayer,
			omittedSelector,			// This selector index will be an average of the other selectors.
			unitFieldsOut + fieldSize*fieldSize*i, 
			selectorFieldsOut + fieldSize*fieldSize*numSelec*i);
	}
}

void makeFields_reg(
	kint * queryReg,
	kint * regArr,
	uint keyLen,
	uint numLoc,
	uint numFields,
	uint stepSize,
	uint fieldSize,
	uint paddingRatio,
	float unitSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitFieldsOut, 
	float *selectorFieldsOut)
{
	regDistPair *wideknn = malloc((stepSize*fieldSize*paddingRatio+1)*sizeof(regDistPair));
	regDistPair *narrowknn = malloc(fieldSize*sizeof(regDistPair));
	kint *knn_regWorkSpace = malloc(keyLen*fieldSize*sizeof(kint));


	makeFields_reg_INTERNAL(
		queryReg,
		regArr,
		keyLen,
		numLoc,
		numFields,
		stepSize,
		fieldSize,
		paddingRatio,
		unitSigma,
		selectorSigma,
		hyperplaneLayer, 
		posSelLayer,
		omittedSelector,			// This selector index will be an average of the other selectors.
		unitFieldsOut, 
		selectorFieldsOut,
		wideknn,
		narrowknn,
		knn_regWorkSpace);
	free(wideknn);
	free(narrowknn);
	free(knn_regWorkSpace);
}

struct makeFields_reg_threadArgs {
	kint ** queryRegs;
	uint numQuery;
	kint * regArr;
	uint keyLen;
	uint numLoc;
	uint numFields;
	uint stepSize;
	uint fieldSize;
	uint paddingRatio;
	double unitSigma;
	double selectorSigma;
	nnLayer * hyperplaneLayer; 
	nnLayer * posSelLayer;
	int omittedSelector;
	float *unitFieldsOut; 
	float *selectorFieldsOut;
	regDistPair *wideknn;
	regDistPair *narrowknn;
	kint * knn_regWorkSpace;
	uint tid;
	uint numThreads;
};

void * batchMakeFields_reg_thread(void *thread_args)
{
	struct makeFields_reg_threadArgs *myargs;
	myargs = (struct makeFields_reg_threadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;
	uint numSelec = myargs->posSelLayer->outDim;
	uint numQuery = myargs->numQuery;
	uint numFields = myargs->numFields;
	uint fieldSize = myargs->fieldSize;
	
	for(uint i = tid; i < numQuery; i += numThreads){
		makeFields_reg_INTERNAL(
			myargs->queryRegs[i],
			myargs->regArr,
			myargs->keyLen,
			myargs->numLoc,
			myargs->numFields,
			myargs->stepSize,
			myargs->fieldSize,
			myargs->paddingRatio,
			myargs->unitSigma,
			myargs->selectorSigma,
			myargs->hyperplaneLayer, 
			myargs->posSelLayer,
			myargs->omittedSelector,
			myargs->unitFieldsOut + i*numFields*fieldSize*fieldSize, 
			myargs->selectorFieldsOut + i*numFields*fieldSize*fieldSize*numSelec,
			myargs->wideknn,
			myargs->narrowknn,
			myargs->knn_regWorkSpace);
	}
	return 0;
}


void batchMakeFields_reg(
	kint ** queryRegs,
	uint numQuery,
	kint * regArr,
	uint keyLen,
	uint numLoc,
	uint numFields,
	uint stepSize,
	uint fieldSize,
	uint paddingRatio,
	float unitSigma,
	float selectorSigma,
	nnLayer * hyperplaneLayer, 
	nnLayer * posSelLayer,
	int omittedSelector,			// This selector index will be an average of the other selectors.
	float *unitFieldsOut, 
	float *selectorFieldsOut,
	int numProc)
{
	regDistPair *wideknn = malloc(numProc*(stepSize*fieldSize*paddingRatio+1)*sizeof(regDistPair));
	regDistPair *narrowknn = malloc(numProc*fieldSize*sizeof(regDistPair));
	kint *knn_regWorkSpace = malloc(numProc*keyLen*fieldSize*sizeof(kint));

	struct makeFields_reg_threadArgs *thread_args = malloc(numProc*sizeof(struct makeFields_reg_threadArgs));
	int rc = 0;
	pthread_t threads[numProc];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(int i=0;i<numProc;i++){
		thread_args[i].numThreads = numProc;
		thread_args[i].tid = i;
		thread_args[i].queryRegs = queryRegs;
		thread_args[i].numQuery = numQuery;
		thread_args[i].regArr = regArr;
		thread_args[i].keyLen = keyLen;
		thread_args[i].numLoc = numLoc;
		thread_args[i].stepSize = stepSize;
		thread_args[i].numFields = numFields;
		thread_args[i].fieldSize = fieldSize;
		thread_args[i].paddingRatio = paddingRatio;
		thread_args[i].unitSigma = unitSigma;
		thread_args[i].selectorSigma = selectorSigma;
		thread_args[i].hyperplaneLayer = hyperplaneLayer;
		thread_args[i].posSelLayer = posSelLayer;
		thread_args[i].omittedSelector = omittedSelector;
		thread_args[i].unitFieldsOut = unitFieldsOut;
		thread_args[i].selectorFieldsOut = selectorFieldsOut;
		thread_args[i].wideknn = wideknn + i*(stepSize*fieldSize*paddingRatio+1);
		thread_args[i].narrowknn = narrowknn + i*fieldSize;
		thread_args[i].knn_regWorkSpace = knn_regWorkSpace + i*fieldSize;


		rc = pthread_create(&threads[i], NULL, batchMakeFields_reg_thread, (void *)&thread_args[i]);
		if (rc){
			printf("Error, unable to create thread\n");
			exit(-1);
		}
	}

	for(int i=0; i < numProc; i++ ){
		rc = pthread_join(threads[i], &status);
		if (rc){
			printf("Error, unable to join: %d \n", rc);
			exit(-1);
     	}
	}
	free(wideknn);
	free(narrowknn);
	free(knn_regWorkSpace);
	free(thread_args);
}

void makeEmbeddedFields_reg_INTERNAL(
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
	float *selectorEigenValsOut,
	regDistPair *wideknn,
	regDistPair *narrowknn,
	float *unitFieldsScratch,
	float *selectorFieldsScratch,
	float *eigenVecScratch,
	float *scratchSpace,
	int *isuppz,
	kint * knn_regWorkSpace)
{
	uint numSelec = posSelLayer->outDim;
	int rc = 0;
	makeFields_reg_INTERNAL(
		queryReg,
		regArr,
		keyLen,
		numLoc,
		numFields,
		stepSize,
		fieldSize,
		paddingRatio,
		unitSigma,
		selectorSigma,
		hyperplaneLayer, 
		posSelLayer,
		omittedSelector,			// This selector index will be an average of the other selectors.
		unitFieldsScratch, 
		selectorFieldsScratch,
		wideknn,
		narrowknn,
		knn_regWorkSpace);

	for (uint i = 0; i < numFields; ++i)
	{	
		c_normalizedLaplacianField(
			fieldSize,
			numSelec,
			unitFieldsScratch + i*fieldSize*fieldSize,
			selectorFieldsScratch + i*fieldSize*fieldSize*numSelec);

		rc = c_spectralEmbedding_internal(
			fieldSize,
			numSelec,
			embeddingSize,
			unitFieldsScratch + i*fieldSize*fieldSize,
			unitEmbeddingsOut + i*fieldSize*embeddingSize,
			unitEigenValsOut + i*fieldSize,
			selectorFieldsScratch + i*fieldSize*fieldSize*numSelec,
			selectorEmbeddingsOut + i*fieldSize*embeddingSize*numSelec,
			selectorEigenValsOut + i*fieldSize,
			scratchSpace,
			eigenVecScratch,
			isuppz);

		if(rc){
			int selectorError = 0;
			if(rc < 0)
				selectorError = rc/(2*fieldSize);
			printf("EigenIssue: matrix %d, value %d\n",selectorError,rc - selectorError*2*fieldSize);
			exit(-1);
		}


	}
}

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
	float *selectorEigenValsOut)
{
	regDistPair *wideknn = malloc((stepSize*fieldSize*paddingRatio + 1)*sizeof(regDistPair));
	regDistPair *narrowknn = malloc(fieldSize*sizeof(regDistPair));
	float *unitFieldsScratch = malloc(numFields*fieldSize*fieldSize*sizeof(float));
	float *selectorFieldsScratch = malloc(numFields*fieldSize*fieldSize*posSelLayer->outDim*sizeof(float));
	float *eigenVecScratch = malloc(fieldSize*fieldSize*sizeof(float));
	float *scratchSpace = malloc(fieldSize*fieldSize*sizeof(float));
	int *isuppz = malloc(fieldSize*2*sizeof(int));
	kint *knn_regWorkSpace = malloc(keyLen*fieldSize*sizeof(kint));

	makeEmbeddedFields_reg_INTERNAL(
			queryReg,
			regArr,
			keyLen,
			numLoc,
			stepSize,
			numFields,
			fieldSize,
			embeddingSize,
			paddingRatio,
			unitSigma,
			selectorSigma,
			hyperplaneLayer, 
			posSelLayer,
			omittedSelector,			// This selector index will be an average of the other selectors.
			unitEmbeddingsOut, 
			unitEigenValsOut,
			selectorEmbeddingsOut,
			selectorEigenValsOut,
			wideknn,
			narrowknn,
			unitFieldsScratch,
			selectorFieldsScratch,
			eigenVecScratch,
			scratchSpace,
			isuppz,
			knn_regWorkSpace);

	free(wideknn);
	free(narrowknn);
	free(unitFieldsScratch);
	free(selectorFieldsScratch);
	free(eigenVecScratch);
	free(scratchSpace);
	free(isuppz);
	free(knn_regWorkSpace);
}

struct makeEmbeddedFields_threadArgs {
	kint ** queryRegs;
	uint numQuery;
	kint * regArr;
	uint keyLen;
	uint numLoc;
	uint stepSize;
	uint numFields;
	uint fieldSize;
	uint embeddingSize;
	uint paddingRatio;
	double unitSigma;
	double selectorSigma;
	nnLayer * hyperplaneLayer; 
	nnLayer * posSelLayer;
	int omittedSelector;
	float *unitEmbeddingsOut; 
	float *unitEigenValsOut;
	float *selectorEmbeddingsOut;
	float *selectorEigenValsOut;
	int *pointTypesOut; 
	int *pointIndexesOut;
	int *pointPopulationOut;

	regDistPair *wideknn;
	regDistPair *narrowknn;
	float *unitFieldsScratch;
	float *selectorFieldsScratch;
	float *eigenVecScratch;
	float *scratchSpace;
	int *isuppz;
	kint * knn_regWorkSpace;

	uint tid;
	uint numThreads;
};

void * batchMakeEmbeddedFields_thread(void *thread_args)
{
	struct makeEmbeddedFields_threadArgs *myargs;
	myargs = (struct makeEmbeddedFields_threadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;
	uint numSelec = myargs->posSelLayer->outDim;
	uint numQuery = myargs->numQuery;
	uint numFields = myargs->numFields;
	uint fieldSize = myargs->fieldSize;
	uint embeddingSize = myargs->embeddingSize;
	
	for(uint i = tid; i < numQuery; i += numThreads){
		makeEmbeddedFields_reg_INTERNAL(
			myargs->queryRegs[i],
			myargs->regArr,
			myargs->keyLen,
			myargs->numLoc,
			myargs->stepSize,
			myargs->numFields,
			myargs->fieldSize,
			myargs->embeddingSize,
			myargs->paddingRatio,
			myargs->unitSigma,
			myargs->selectorSigma,
			myargs->hyperplaneLayer, 
			myargs->posSelLayer,
			myargs->omittedSelector,
			myargs->unitEmbeddingsOut + i*numFields*fieldSize*embeddingSize, 
			myargs->unitEigenValsOut + i*numFields*fieldSize,
			myargs->selectorEmbeddingsOut + i*numFields*fieldSize*embeddingSize*numSelec,
			myargs->selectorEigenValsOut + i*numFields*fieldSize*numSelec,
			myargs->wideknn,
			myargs->narrowknn,
			myargs->unitFieldsScratch,
			myargs->selectorFieldsScratch,
			myargs->eigenVecScratch,
			myargs->scratchSpace,
			myargs->isuppz,
			myargs->knn_regWorkSpace);
	}
	return 0;
}


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
	int numProc)
{
	regDistPair *wideknn = malloc(numProc*(numFields*stepSize*fieldSize*paddingRatio + 1)*sizeof(regDistPair));
	regDistPair *narrowknn = malloc(numProc*fieldSize*sizeof(regDistPair));
	float *unitFieldsScratch = malloc(numProc*numFields*fieldSize*fieldSize*sizeof(float));
	float *selectorFieldsScratch = malloc(numProc*numFields*fieldSize*fieldSize*posSelLayer->outDim*sizeof(float));
	float *eigenVecScratch = malloc(numProc*fieldSize*fieldSize*sizeof(float));
	float *scratchSpace = malloc(numProc*fieldSize*fieldSize*sizeof(float));
	int *isuppz = malloc(numProc*fieldSize*2*sizeof(int));
	kint *knn_regWorkSpace = malloc(numProc*keyLen*fieldSize*sizeof(kint));

	struct makeEmbeddedFields_threadArgs *thread_args = malloc(numProc*sizeof(struct makeEmbeddedFields_threadArgs));
	int rc = 0;
	pthread_t threads[numProc];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(int i=0;i<numProc;i++){
		thread_args[i].numThreads = numProc;
		thread_args[i].tid = i;
		thread_args[i].queryRegs = queryRegs;
		thread_args[i].numQuery = numQuery;
		thread_args[i].regArr = regArr;
		thread_args[i].keyLen = keyLen;
		thread_args[i].numLoc = numLoc;
		thread_args[i].stepSize = stepSize;
		thread_args[i].numFields = numFields;
		thread_args[i].fieldSize = fieldSize;
		thread_args[i].embeddingSize = embeddingSize;
		thread_args[i].paddingRatio = paddingRatio;
		thread_args[i].unitSigma = unitSigma;
		thread_args[i].selectorSigma = selectorSigma;
		thread_args[i].hyperplaneLayer = hyperplaneLayer;
		thread_args[i].posSelLayer = posSelLayer;
		thread_args[i].omittedSelector = omittedSelector;
		thread_args[i].unitEmbeddingsOut = unitEmbeddingsOut;
		thread_args[i].unitEigenValsOut = 	unitEigenValsOut;
		thread_args[i].selectorEmbeddingsOut = selectorEmbeddingsOut;
		thread_args[i].selectorEigenValsOut = selectorEigenValsOut;
		
		thread_args[i].wideknn = wideknn + i*stepSize*fieldSize*paddingRatio;
		thread_args[i].narrowknn = narrowknn + i*fieldSize;
		thread_args[i].unitFieldsScratch = unitFieldsScratch + i*fieldSize*fieldSize*numFields;
		thread_args[i].selectorFieldsScratch = 	selectorFieldsScratch + i*numFields*fieldSize*fieldSize*posSelLayer->outDim;
		thread_args[i].eigenVecScratch = eigenVecScratch + i*fieldSize*fieldSize;
		thread_args[i].scratchSpace = scratchSpace + i*fieldSize*fieldSize;
		thread_args[i].isuppz = isuppz + i*2*fieldSize;
		thread_args[i].knn_regWorkSpace = knn_regWorkSpace + i*fieldSize;

		rc = pthread_create(&threads[i], NULL, batchMakeEmbeddedFields_thread, (void *)&thread_args[i]);
		if (rc){
			printf("Error, unable to create thread\n");
			exit(-1);
		}
	}

	for(int i=0; i < numProc; i++ ){
		rc = pthread_join(threads[i], &status);
		if (rc){
			printf("Error, unable to join: %d \n", rc);
			exit(-1);
     	}
	}
	free(wideknn);
	free(narrowknn);
	free(thread_args);
	free(unitFieldsScratch);
	free(selectorFieldsScratch);
	free(eigenVecScratch);
	free(scratchSpace);
	free(isuppz);


}