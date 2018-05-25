#include "coo_graphTools.h"
#include <math.h>

GraphAttr * allocateGraphAttr( 
	uint numNodes, 
	uint numSelectors, 
	uint numHps,
	float unitalSigma, 
	float selectorSigma,
	uint pathThreshold)
{
	GraphAttr * ngs = malloc(sizeof(GraphAttr));

	ngs->pointTypes = malloc(numNodes*sizeof(int));
	ngs->pointIndexes = malloc(numNodes*sizeof(int));

	ngs->edgesI = malloc(numNodes*numNodes*sizeof(int));
	ngs->edgesJ = malloc(numNodes*numNodes*sizeof(int));

	ngs->unitalWeights = malloc(numNodes*numNodes*sizeof(int));
	ngs->selectorWeights = malloc(numNodes*numNodes*numSelectors*sizeof(int));
	
	ngs->unitalSigma = unitalSigma;
	ngs->selectorSigma = selectorSigma;
	ngs->pathThreshold = pathThreshold;

	ngs->numNodes = numNodes;
	ngs->numSelectors = numSelectors;
	ngs->numHps = numHps;
	
	ngs->edgeIndex = 0;
	int rc = pthread_spin_init(&(ngs->edgeIndexSpinlock), 0);
	if(rc){
		printf("allocare graph spinlock failed\n");
		//printMatrix(tc->layer->A,n,m);
		exit(-1);
	}
	return ngs;
}


void freeGraphAttr(GraphAttr * gts)
{
	if(gts){
		free(gts->pointTypes);
		free(gts->pointIndexes);
		free(gts->edgesI);
		free(gts->edgesJ);

		free(gts->unitalWeights);
		free(gts->selectorWeights);
		free(gts);
	}
}

struct gtsThreadArgs {
	uint tid;
	uint numThreads;
	location ** locArr;
	uint numLoc;

	GraphAttr * gts;
	nnLayer * posSelLayer;
};

void * c_getGraphAttr_thread(void *thread_args)
{

	struct gtsThreadArgs *myargs;
	myargs = (struct gtsThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	location ** locArr = myargs->locArr;
	uint numLoc = myargs->numLoc;

	GraphAttr * graphAttr = myargs->gts;

	uint keyLen = calcKeyLen(graphAttr->numHps);
	uint numSelec = graphAttr->numSelectors;

	kint * key_i;
	kint * key_j;

	float unitalSigma = graphAttr->unitalSigma;
	float selectorSigma = graphAttr->selectorSigma;

	int pathThreshold = graphAttr->pathThreshold;

	float unitalWeight;

	int myEdgeIndex;

	for(uint i = tid; i < numLoc; i += numThreads){
		key_i = locArr[i]->regKey;

		graphAttr->pointTypes[i] = locArr[i]->loc.pointTypes[0];
		graphAttr->pointIndexes[i] = locArr[i]->loc.pointIndexes[0];


		for(uint j = i+1; j < numLoc; j++){
			key_j = locArr[j]->regKey;
			unitalWeight = numberOfDiff(key_i,key_j, keyLen);
			if(unitalWeight <= pathThreshold){

				pthread_spin_lock(&(graphAttr->edgeIndexSpinlock));
					myEdgeIndex = graphAttr->edgeIndex;
					graphAttr->edgeIndex++;
				pthread_spin_unlock(&(graphAttr->edgeIndexSpinlock));


				graphAttr->edgesI[myEdgeIndex] = i;
				graphAttr->edgesJ[myEdgeIndex] = j;
				
				graphAttr->unitalWeights[myEdgeIndex] = exp(-(unitalWeight)/unitalSigma);
				getSelectorWeights(key_i,key_j,myargs->posSelLayer,graphAttr->selectorWeights+myEdgeIndex*numSelec);
				for(uint k = 0; k < numSelec; k++){
					graphAttr->selectorWeights[k + numSelec*myEdgeIndex] = exp(-graphAttr->selectorWeights[k + numSelec*myEdgeIndex]/selectorSigma);
				}
			}
		}
	}

	return 0;
}

GraphAttr * c_getGraphAttr(
	location ** locArr,
	int numLoc,
	const double unitalSigma,
	const double selectorSigma,
	nnLayer * hyperplaneLayer,
	nnLayer * selectorLayer,
	int pathThreshold,
	int numProc)
{
	
	nnLayer * posSelLayer = copyLayerPositive(selectorLayer,0);

	GraphAttr * graphAttr = allocateGraphAttr( 
		numLoc, 
		selectorLayer->outDim,
		hyperplaneLayer->outDim, 
		unitalSigma, 
		selectorSigma,
		pathThreshold);

	struct gtsThreadArgs *thread_args = malloc(numProc*sizeof(struct gtsThreadArgs));

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
		thread_args[i].gts = graphAttr;
		thread_args[i].posSelLayer = posSelLayer;

		rc = pthread_create(&threads[i], NULL, c_getGraphAttr_thread, (void *)&thread_args[i]);
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
	return(graphAttr);
}



void gaussianSimilarity(float * weights, int count, float sigma)
{
	//int minIndex = cblas_isamin (count, weights, 1);
	//float minVal = weights[minIndex];
	for(int i = 0; i<count;i++){
		weights[i] = exp(-(weights[i]*weights[i])/sigma);
	}
}

float gaussianSimilarity1(float weight,float sigma)
{
	return exp(-(weight*weight)/sigma);
}

int getEdgeIndex(GraphAttr * gts)
{
	return gts->edgeIndex;
}

void getEdgeAttr(GraphAttr * gts, float * unitalWeightsOut, float *selectorWeightsOut)
{
    memcpy(unitalWeightsOut,gts->unitalWeights,gts->edgeIndex *sizeof(float));
    memcpy(selectorWeightsOut,gts->selectorWeights,gts->edgeIndex * gts->numSelectors *sizeof(float));
} 

void getEdgesCOO(GraphAttr * gts, int * edgesIout, int *edgesJout)
{
    memcpy(edgesIout,gts->edgesI,gts->edgeIndex*sizeof(int));
    memcpy(edgesJout,gts->edgesJ,gts->edgeIndex*sizeof(int));
}

void getEdges(GraphAttr * gts, int * edgesOut)
{
	for (int i = 0; i < gts->edgeIndex; ++i)
	{
		edgesOut[2*i] = gts->edgesI[i];
		edgesOut[2*i+1] = gts->edgesJ[i];
	}
}

void getVertexAttr(GraphAttr *gts, int *pointTypesOut,int *pointIndexesOut)
{
	memcpy(pointTypesOut,gts->pointTypes,gts->numNodes*sizeof(int));
	memcpy(pointIndexesOut,gts->pointIndexes,gts->numNodes*sizeof(int));
}


void printEdgeAttr(GraphAttr *gts)
{
	for (int i = 0; i < gts->edgeIndex; ++i)
	{
		printf("Edge %d (%d,%d): U: %f, ", i,gts->edgesI[i],gts->edgesJ[i],gts->unitalWeights[i]);
		for (uint k = 0; k < gts->numSelectors-1; ++k)
		{
			printf("S(%d): %f,",k,gts->selectorWeights[i*gts->numSelectors + k]);
		}
		printf("S(%d): %f \n",gts->numSelectors-1,gts->selectorWeights[i*gts->numSelectors + gts->numSelectors-1]);
	}
}