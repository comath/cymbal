#include "csr_graphTools.h"

#define INITCAP 8

void printCSR(SparseGraphMatrix *csr)
{
	for (uint i = 0; i < csr->numNodes; ++i)
	{
		printf("IA[%d]: %u\n",i, csr->IA[i]);
		printf("Idiag[%d]: %u\n",i,csr->Idiag[i]);

		for (uint j = csr->IA[i]; j < csr->IA[i+1]; ++j)
		{
			printf("A(%d,%d): %f,        ",i,csr->JA[j],csr->unitWeights[j]);
		}
		printf("\n");
	}

	printf("nnz: %lu\n", csr->nnz);
	for(uint i = 0; i<csr->nnz; ++i){
		printf("JA[%d]: %u\n",i,csr->JA[i]);
		printf("data[%d]: %f\n",i,csr->unitWeights[i]);

		if(csr->selectorWeights){
			for(uint j = 0; j < csr->numSel; ++j){
				printf("dataSel[%d][%d]: %f\n",i,j,csr->selectorWeights[i*csr->numSel+j]);
			}
		}
	}
}

struct csrPackingThreadArgs {
	uint tid;
	uint numThreads;
	location ** locArr;
	uint numLoc;
	uint keyLen;
	csrStaging *csrS;

	uint threadNodeCount;
};


static void row_resize(struct csrStaging *csrS, int capacity, int row)
{
    csrS->rowColumns[row] = realloc(csrS->rowColumns[row], sizeof(uint) * capacity);
    csrS->rowData[row] = realloc(csrS->rowData[row], sizeof(float) * capacity);
    csrS->rowCapacity[row] = capacity;
}

csrStaging * allocate_csrStaging( 
	uint numNodes,  
	int pathThreshold,
	float unitSigma,
	uint numHps)
{
	csrStaging * csrS = malloc(sizeof(csrStaging));
	csrS->pathThreshold = pathThreshold;
	csrS->unitSigma = unitSigma;
	csrS->numHps = numHps;
	csrS->numNodes = numNodes;

	csrS->rowCount = malloc(numNodes*sizeof(uint));
	csrS->rowDiagIndex = malloc(numNodes*sizeof(uint));
	csrS->rowCapacity = malloc(numNodes*sizeof(uint));
	csrS->rowColumns = malloc(numNodes*sizeof(uint *));
	csrS->rowData = malloc(numNodes*sizeof(float *));
	csrS->rowDiagData = malloc(numNodes*sizeof(float));


	for(uint i = 0; i<numNodes;++i){
		csrS->rowCount[i] = 0;
		csrS->rowCapacity[i] = INITCAP;
		csrS->rowColumns[i] = malloc(INITCAP*sizeof(uint));
		csrS->rowData[i] = malloc(INITCAP*sizeof(float));
	}
	return csrS;
}

void free_csrStaging(csrStaging * csrS)
{
	for(uint i = 0; i<csrS->numNodes;++i){
		free(csrS->rowColumns[i]);
		free(csrS->rowData[i]);
	}
	free(csrS->rowCount);
	free(csrS->rowCapacity);
	free(csrS->rowColumns);
	free(csrS->rowData);
}

int getMaxDegreeIndex(int *indexes, float *degrees, uint k)
{
	int maxDegreeIndex = 0;
	float maxDegree = degrees[0];
	for(uint i = 0; i < k; ++i){
		if(degrees[indexes[i]] > maxDegree){
			maxDegree = degrees[indexes[i]];
			maxDegreeIndex = indexes[i];
		}
	}
	return maxDegreeIndex;
}

int intOrder(const void * a, const void * b)
{
	const int *myA = a;
	const int *myB = b;
	//printf("Comparing %d with %d\n", myA->graphDist , myB->graphDist);

	return *myA - *myB;
}

void smallestKDegreeNodes(csrStaging *csrS, uint k, int *lowDegreeIndexes)
{
	int maxDegreeIndex = 0;
	float maxDegree = -1.0;

	for(uint i = 0; i < k; ++i){
		lowDegreeIndexes[i] = i;
		if(csrS->rowDiagData[i] > maxDegree){
			maxDegreeIndex = i;
			maxDegree = csrS->rowDiagData[i];
		}
	}

	for(uint i = k; i < csrS->numNodes; ++i){
		if(csrS->rowDiagData[i] < maxDegree){
			lowDegreeIndexes[maxDegreeIndex] = i;
			maxDegreeIndex = getMaxDegreeIndex(lowDegreeIndexes, csrS->rowDiagData, k);
			maxDegree = csrS->rowDiagData[maxDegreeIndex];
		}
	}
	
	qsort(lowDegreeIndexes, k, sizeof(int), intOrder);
}

void removeRowFromStaging(csrStaging * csrS,int rowIndex)
{

	csrS->nnz -= csrS->rowCount[rowIndex];
	free(csrS->rowColumns[rowIndex]);
	free(csrS->rowData[rowIndex]);

	for(uint i = rowIndex; i < csrS->numNodes - 1; i++){

		csrS->rowCount[i] = csrS->rowCount[i+1];
		csrS->rowDiagIndex[i] = csrS->rowDiagIndex[i+1];
		csrS->rowCapacity[i] = csrS->rowCapacity[i+1];
		csrS->rowColumns[i] = csrS->rowColumns[i+1];
		csrS->rowData[i] = csrS->rowData[i+1];
		csrS->rowDiagData[i] = csrS->rowDiagData[i+1];
	}
	csrS->numNodes--;
}

void removeColFromStagingRow(csrStaging * csrS, int rowIndex,int columnIndex)
{
	int matchingIndex = -1;
	for(uint i = rowIndex; i < csrS->rowCount[rowIndex]; i++){
		if(csrS->rowColumns[rowIndex][i] > columnIndex){
			csrS->rowColumns[rowIndex][i]--;
		} else {
			if(csrS->rowColumns[rowIndex][i] == columnIndex){
				matchingIndex = i;
			}
		}
	}
	if(matchingIndex != -1){
		for(int i = matchingIndex; i < csrS->rowCount[rowIndex] - 1; i++){
			csrS->rowData[rowIndex][i] = csrS->rowData[rowIndex][i+1];
			csrS->rowColumns[rowIndex][i] = csrS->rowColumns[rowIndex][i+1];
		}
		csrS->rowCount[rowIndex]--;
	}
}

void removeLowDegree(csrStaging *csrS, int numToRemove)
{
	int * lowDegreeIndexes = malloc(numToRemove*sizeof(float));
	smallestKDegreeNodes(csrS, numToRemove, lowDegreeIndexes);
	for(int i = numToRemove-1; i > -1; i--){
		removeRowFromStaging(csrS,lowDegreeIndexes[i]);
	}
	for(uint i = 0; i < csrS->numNodes; i++){
		for(int j = numToRemove-1; j > -1; j--){
			removeColFromStagingRow(csrS, i, lowDegreeIndexes[j]);
		}
	}
}

SparseGraphMatrix * allocate_UnitCSR( 
	uint numNodes,  
	unsigned long int nnz,
	uint *IA,
	uint *JA,
	uint *Idiag,
	float *unitWeights,
	float *unitDiag)
{
	SparseGraphMatrix * csrU = malloc(sizeof(SparseGraphMatrix));
	
	csrU->nnz = nnz;
	csrU->numNodes = numNodes;
	csrU->IA = IA;
	csrU->Idiag = Idiag;
	csrU->JA = JA;
	csrU->unitWeights = unitWeights;
	csrU->unitDiagWeights = unitDiag;
	csrU->selectorWeights = NULL;
	csrU->numSel = 0;
	return csrU;
}

SparseGraphMatrix * allocate_CSR( 
	uint numNodes,  
	unsigned long int nnz,
	uint *IA,
	uint *JA,
	uint *Idiag,
	float *unitWeights,
	float *unitDiag,
	float *selectorWeights,
	float *selectorDiag)
{
	SparseGraphMatrix * csrU = allocate_UnitCSR( 
		numNodes,  
		nnz,
		IA,
		JA,
		Idiag,
		unitWeights,
		unitDiag);
	csrU->selectorWeights = selectorWeights;
	csrU->selectorDiagWeights = selectorDiag;
	return csrU;
}

void populateUnitCSR(
	csrStaging *csrS,
	SparseGraphMatrix * csrU)
{
	uint indexIA = 0;
	for(uint i = 0; i < csrU->numNodes; ++i){
		csrU->IA[i] = indexIA;
		csrU->Idiag[i] = indexIA + csrS->rowDiagIndex[i];
		memcpy(csrU->unitWeights + indexIA,csrS->rowData[i],csrS->rowCount[i]*sizeof(float));
		memcpy(csrU->JA + indexIA,csrS->rowColumns[i],csrS->rowCount[i]*sizeof(uint));
		csrU->unitDiagWeights[i] = csrS->rowDiagData[i];
		indexIA += csrS->rowCount[i];
	}
	csrU->IA[csrU->numNodes] = indexIA;
}

void * c_getStagingCSR_thread(void *thread_args)
{

	struct csrPackingThreadArgs *myargs;
	myargs = (struct csrPackingThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	location ** locArr = myargs->locArr;
	uint numLoc = myargs->numLoc;

	struct csrStaging *csrS = myargs->csrS;

	uint keyLen = calcKeyLen(csrS->numHps);
	float unitSigma = csrS->unitSigma;
	kint * key_i;

	int pathThreshold = csrS->pathThreshold;
	uint curRowCount = 0;
	float unitalWeight;
	myargs->threadNodeCount = 0;

	for(uint i = tid; i < numLoc; i += numThreads){
		key_i = locArr[i]->regKey;
		curRowCount = 0;
		csrS->rowDiagData[i] = 0;
		for(uint j = 0; j < numLoc; j++){
			if(i==j){
				if(curRowCount == csrS->rowCapacity[i])
					row_resize(csrS,csrS->rowCapacity[i] << 1,i);
				csrS->rowDiagIndex[i] = curRowCount;
				csrS->rowData[i][curRowCount] = 0;
				csrS->rowColumns[i][curRowCount] = i;
				curRowCount++;
			} else {
				unitalWeight = numberOfDiff(key_i,locArr[j]->regKey, keyLen);
				if(unitalWeight <= pathThreshold){
					if(curRowCount == csrS->rowCapacity[i])
						row_resize(csrS,csrS->rowCapacity[i] << 1,i);
					csrS->rowColumns[i][curRowCount] = j;
					if(unitSigma > 0)
						csrS->rowData[i][curRowCount] = exp(-unitalWeight/unitSigma);
					else
						csrS->rowData[i][curRowCount] = unitalWeight;
					csrS->rowDiagData[i] += csrS->rowData[i][curRowCount];
					curRowCount++;
				}
			}
		}
		csrS->rowCount[i] = curRowCount;
		myargs->threadNodeCount += curRowCount;
	}
	return 0;
}

csrStaging * c_getStagingCSR(
	location ** locArr,
	int numLoc,
	int numHp,
	float unitSigma,
	int pathThreshold,
	int numProc)
{
	csrStaging * csrS = allocate_csrStaging(numLoc,pathThreshold,unitSigma,numHp);

	struct csrPackingThreadArgs *thread_args = malloc(numProc*sizeof(struct csrPackingThreadArgs));

	int rc;

	pthread_t threads[numProc];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(int i=0;i<numProc;i++){
		thread_args[i].numThreads = numProc;
		thread_args[i].tid = i;
		thread_args[i].locArr = locArr;
		thread_args[i].numLoc = numLoc;
		thread_args[i].csrS = csrS;

		rc = pthread_create(&threads[i], NULL, c_getStagingCSR_thread, (void *)&thread_args[i]);
		if (rc){
			printf("Error, unable to create thread\n");
			exit(-1);
		}
	}
	csrS->nnz = 0;

	for(int i=0; i < numProc; i++ ){
		rc = pthread_join(threads[i], &status);
		if (rc){
			printf("Error, unable to join: %d \n", rc);
			exit(-1);
     	}
     	csrS->nnz += thread_args[i].threadNodeCount;
	}

	free(thread_args);
	return(csrS);
}



struct populateCSRThreadArgs {
	uint tid;
	uint numThreads;
	location ** locArr;
	csrStaging *csrS;
	SparseGraphMatrix *csr;
	nnLayer * posSelLayer;
};

void * c_populateCSR_thread(void *thread_args)
{
	struct populateCSRThreadArgs *myargs = thread_args;
	csrStaging *csrS = myargs->csrS;
	SparseGraphMatrix *csr = myargs->csr;
	nnLayer * posSelLayer = myargs->posSelLayer;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;
	uint indexIA = 0;
	uint i = 0;
	uint j = 0;
	uint k = 0;
	uint numSel = csr->numSel;
	unsigned long int nnz = csrS->nnz;
	float selectorSigma = csr->selectorSigma;
	kint *key_i;

	location **locArr = myargs->locArr;

	uint numNodes = csr->numNodes;

	for(i = 0;i<tid;i++){
		indexIA += csrS->rowCount[i];
	}
	for(i = tid; i < numNodes; i+= numThreads){
		
		csr->IA[i] = indexIA;
		memcpy(csr->JA + indexIA,csrS->rowColumns[i],csrS->rowCount[i]*sizeof(uint));
		
		csr->Idiag[i] = indexIA + csrS->rowDiagIndex[i];
		csr->unitDiagWeights[i] = csrS->rowDiagData[i];
		
		memcpy(csr->unitWeights + indexIA,csrS->rowData[i],csrS->rowCount[i]*sizeof(float));
		
		key_i = locArr[i]->regKey;
		for(k = 0; k < numSel; ++k){
			csr->selectorWeights[indexIA + csrS->rowDiagIndex[i] + k*nnz] = 0;
			csr->selectorDiagWeights[i + k*numNodes] = 0;
		}

		for(j = 0; j < csrS->rowCount[i]; ++j){
			
			if(j != csrS->rowDiagIndex[i]){
				getSelectorWeightsT(
					key_i,
					locArr[csrS->rowColumns[i][j]]->regKey,
					posSelLayer,
					csr->selectorWeights + (indexIA + j),
					nnz);
				for(k = 0; k < numSel; ++k){
					if(selectorSigma > 0)
						csr->selectorWeights[(indexIA + j) + k*nnz] = exp(-csr->selectorWeights[(indexIA + j) + k*nnz]/selectorSigma);
					else
						csr->selectorWeights[(indexIA + j) + k*nnz] = csr->selectorWeights[(indexIA + j) + k*nnz];
					csr->selectorDiagWeights[i + k*numNodes] += csr->selectorWeights[(indexIA + j) + k*nnz];
				}
			} 
		}
			
		for(j = i; j < i+numThreads && j < numNodes;j++){
			indexIA += csrS->rowCount[j];
		}
	}
	if(tid == 0){
		csr->IA[numNodes] = indexIA;
	}
	return 0;
}


void populateCSR(
	location ** locArr,
	nnLayer *posSelLayer,
	float selectorSigma,
	csrStaging *csrS,
	SparseGraphMatrix * csr,
	int numProc)
{
	struct populateCSRThreadArgs *thread_args = malloc(numProc*sizeof(struct populateCSRThreadArgs));

	csr->numSel = posSelLayer->outDim;
	csr->selectorSigma = selectorSigma;
	int rc;

	pthread_t threads[numProc];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(int i=0;i<numProc;i++){
		thread_args[i].numThreads = numProc;
		thread_args[i].tid = i;
		thread_args[i].locArr = locArr;
		thread_args[i].csrS = csrS;
		thread_args[i].csr = csr;
		thread_args[i].posSelLayer = posSelLayer;


		rc = pthread_create(&threads[i], NULL, c_populateCSR_thread, (void *)&thread_args[i]);
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
	free(thread_args);	
}

struct normalizedLaplacianCSRThreadArgs {
	uint tid;
	uint numThreads;
	double *unitDiagSqrt;
	double *selectorDiagSqrt;
	SparseGraphMatrix *csr;
	float *unitLap;
	float *selectorLap;
};

void * c_normalizeLaplacian_thread(void *thread_args)
{
	struct normalizedLaplacianCSRThreadArgs *myargs = thread_args;
	SparseGraphMatrix *csr = myargs->csr;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;
	uint i = 0;
	uint j = 0;
	uint k = 0;

	uint numSel = csr->numSel;
	uint numNodes = csr->numNodes;

	float *unitLap = myargs->unitLap;
	float *selectorLap = myargs->selectorLap;
	double *unitDiagSqrt = myargs->unitDiagSqrt;
	double *selectorDiagSqrt = myargs->selectorDiagSqrt;

	for(i = tid; i < numNodes; i+= numThreads){	
		for(j = csr->IA[i]; j < csr->IA[i+1]; ++j){
			if(j != csr->Idiag[i]){
				unitLap[j] = -(csr->unitWeights[j] * (unitDiagSqrt[i] * unitDiagSqrt[csr->JA[j]]));
				
				for(k = 0; k < numSel; ++k){
					selectorLap[j*numSel + k] = -(csr->selectorWeights[j*numSel + k] / (selectorDiagSqrt[i*numSel + k] * selectorDiagSqrt[csr->JA[j]*numSel + k]));
				}
			} else {
				unitLap[j] = 1;
				for(k = 0; k < numSel; ++k){
					selectorLap[j*numSel + k] = 1;
				}
			}
		}
	}
	return 0;
}

void c_normalizedLaplacianCSR(SparseGraphMatrix * csr, float *unitLap, float *selectorLap, int numProc)
{
	struct normalizedLaplacianCSRThreadArgs *thread_args = malloc(numProc*sizeof(struct normalizedLaplacianCSRThreadArgs));

	// Create sqrt(D)^(-1)
	double * unitDiagSqrt = malloc(csr->numNodes*sizeof(double));
	double * selectorDiagSqrt = malloc(csr->numNodes*csr->numSel*sizeof(double));
	for (uint i = 0; i < csr->numNodes; ++i)
	{
		unitDiagSqrt[i] = 1.0/sqrt(csr->unitDiagWeights[i]);
		for(uint j = 0; j < csr->numSel; j++){
			selectorDiagSqrt[i*csr->numSel + j] = 1.0/sqrt(csr->selectorDiagWeights[i*csr->numSel + j]);
		}
	
	}
	int rc;

	pthread_t threads[numProc];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(int i=0;i<numProc;i++){
		thread_args[i].numThreads = numProc;
		thread_args[i].tid = i;
		thread_args[i].csr = csr;
		thread_args[i].unitDiagSqrt = unitDiagSqrt;
		thread_args[i].selectorDiagSqrt = selectorDiagSqrt;
		thread_args[i].unitLap = unitLap;
		thread_args[i].selectorLap = selectorLap;


		rc = pthread_create(&threads[i], NULL, c_normalizeLaplacian_thread, (void *)&thread_args[i]);
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
	free(unitDiagSqrt);
	free(selectorDiagSqrt);
	free(thread_args);
}


void getSparseVertexAttr(location **loc, uint numLoc, int *pointTypesOut, int *pointIndexesOut)
{
	for(uint i = 0; i < numLoc; i++){
		pointIndexesOut[i] = loc[i]->loc.pointIndexes[0];
		pointTypesOut[i] = loc[i]->loc.pointTypes[0];
	}
}

int searchRange(uint * range, uint start, uint stop, uint target)
{
	for(uint i = start; i < stop; ++i){
		if(range[i] == target){
			return i;
		}
	}
	return -1;
}

int checkSymCSR(SparseGraphMatrix * csr, char verbose)
{
	uint i_index = 0;
	uint j_index = 0;
	int k = 0;
	uint *JA = csr->JA;
	uint *IA = csr->IA;
	uint numSel = csr->numSel;

	int returnVal = 0;
	for(uint i = 0; i < csr->numNodes; i++){	
		for(uint j = IA[i]; j < IA[i+1]; ++j){
			j_index = JA[j];
			i_index = i;
			k = searchRange( JA, IA[j_index],IA[j_index+1], i_index);
			if(k != -1){
				if(i_index <= j_index && csr->unitWeights[j] != csr->unitWeights[k]){
					if(verbose)
						printf("[%d] U(%d,%d) = %f != %f = U(%d,%d) [%d] differ by %f\n",
							j,
							i_index,
							j_index,
							csr->unitWeights[j],
							csr->unitWeights[k],
							j_index,
							i_index,
							k,
							1000000*(csr->unitWeights[j] - csr->unitWeights[k]));
					returnVal++;
				}
				for(uint l = 0; l < numSel; ++l){
					if(i_index <= j_index && csr->selectorWeights[j*numSel + l] != csr->selectorWeights[k*numSel + l]){
						if(verbose)
							printf("[%d] S_%d(%d,%d) = %f != %f = S_%d(%d,%d) [%d] differ by %f\n",
								j,
								l,
								i_index,
								j_index,
								csr->selectorWeights[j*numSel + l],
								csr->selectorWeights[k*numSel + l],
								l,
								j_index,
								i_index,
								k,
								1000000*(csr->selectorWeights[j*numSel + l] - csr->selectorWeights[k*numSel + l]));
						returnVal++;
					}
				}
			} else {
				printf("[%d] U(%d,%d) is not paired with another non-zero\n",j,i_index,j_index);
				returnVal++;
			}
		}
	}
	return returnVal;
}