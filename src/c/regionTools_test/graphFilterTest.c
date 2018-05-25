/*
This just tests that there's no race conditions, or memory leaks. 
Running Valgrind or gdb on the python wrap leads to a bit more headaches.
*/
#include <stdio.h>
#include <stdlib.h>
#include "../cutils/nnLayerUtils.h"
#include "../cutils/key.h"
#include "../cutils/graphTools/local_graphTools.h"
#include "../cutils/graphTools/dence_graphTools.h"






void printFloatArrNoNewLine(float * arr, int numElements){
	int i = 0;
	printf("[");
	for(i=0;i<numElements-1;i++){
		printf("%f,",arr[i]);
	}
	printf("%f", arr[numElements-1]);
	printf("]");
}

//Creates a layer with HPs parrallel to the cordinate axes
nnLayer *createDumbLayer(uint dim, uint numHP)
{
	nnLayer * layer = calloc(1,sizeof(nnLayer));
	layer->A = calloc((dim+1)*numHP,sizeof(float));
	layer->b = layer->A + dim*numHP;
	uint i,j;
	for(i=0;i<numHP;i++){
		for(j=0;j<dim;j++){
			if(i==j){
				layer->A[i*dim + j] = 1;
			} else {
				layer->A[i*dim + j] = 0;
			}
		}
		layer->b[i] = 0;
	}
	layer->inDim = dim;
	layer->outDim = numHP;
	return layer;
}

//Creates a layer with HPs parrallel to the cordinate axes
nnLayer *createDumbSelectionLayer(uint numHP, uint selection)
{
	nnLayer * layer = calloc(1,sizeof(nnLayer));
	layer->A = calloc((numHP+1)*selection,sizeof(float));
	layer->b = layer->A + numHP*selection;
	uint i,j;
	for(i=0;i<selection;i++){
		for(j=0;j<numHP;j++){
			if(i==j){
				layer->A[i*numHP + j] = i + j;
			} else {
				layer->A[i*numHP + j] = i + j;
			}
		}
		layer->b[i] = 0.5-(float)numHP;
	}
	layer->inDim = numHP;
	layer->outDim = selection;
	return layer;
}

void randomizePoint(float *p, uint dim)
{
	uint i = 0;
	for(i=0;i<dim;i++){
		p[i] = (double)((rand() % 20000) - 10000)/10000;
	}
}

float * randomData(uint dim, uint numData)
{
	float * data = malloc(dim*numData*sizeof(float));
	uint i;
	for(i=0;i<numData;i++){
		randomizePoint(data +dim*i,dim);
	}
	return data;
}

void printIntArr(int *arr, uint length){
	uint i = 0;
	printf("[");
	for(i=0;i<length-1;i++){
		printf("%d,",arr[i]);
	}
	printf("%d", arr[length-1]);
	printf("]\n");
}

void fillRegions(kint *regions, int numRegions)
{
	int j;
	regions[0] = 0;
	for (int i = 0; i < numRegions; ++i)
	{
		regions[i] = 0;
		for(j = 0; j< i; ++j){
			if(i & (1 << j))
				addIndexToKey(regions + i, i & (1 << j));
		}
	}
}

int main(int argc, char* argv[])
{
	uint dim = 10;
	uint numHP = 20;
	uint finalDim = 2;

	uint numData = 20;
	uint keySize = calcKeyLen(numHP);

	kint regions[20];

	fillRegions(regions,20);
	printf("20 regions:\n");
	for(int i = 0; i < 20; i++){
		printf("[%d]:\t", i);
		printKey(regions + i,32);
	}
	printf("10 nearest neigbors:\n");
	regDistPair knn[10];
	knn[0].regIndex = -1;
	knn[0].dist = 0;
	knn[0].reg = regions;
	regKNN(regions, 1, regions + 1, 20 - 1, 9, knn + 1);
	for (int i = 0; i < 10; ++i)
	{	
		printf("[%d] Dist: %d, index: %d \t",i,knn[i].dist,knn[i].regIndex);
		printKey(knn[i].reg,32);
	}

	srand(time(NULL));
	printf("If no faliures are printed then we are fine.\n");
	nnLayer *layer0 = createDumbLayer(dim,numHP);
	nnLayer *layer1 = createDumbSelectionLayer(numHP, finalDim);

	printf("Calculating the signature of %d Points\n",numData);
	uint stepSize = 2;
	uint numFields = 5;
	uint fieldSize = 4;
	int numTestPoints = 10;

	float sigma = -1.0;
	float * selectorFieldsOut = malloc(numFields*fieldSize*fieldSize*finalDim*sizeof(float));
	float * selectorAdj = malloc(numData*numData*finalDim*sizeof(float));
	float * unitFieldsOut = malloc(numFields*fieldSize*fieldSize*sizeof(float));
	float * unitAdj = malloc(numData*numData*sizeof(float));

	c_getAdjacancyMatrix_reg(
		regions,
		numData,
		sigma,
		sigma*3,
		1, 
		layer1,
		-1,			// This selector index will be an average of the other selectors.
		unitAdj, 
		selectorAdj);

	printf("\nUnit Adjacency:\n");
	printMatrix(unitAdj,numData,numData);
	for (int i = 0; i < finalDim; ++i){
		printf("\nSelector Adjacency %d:\n", i);
		printMatrix(selectorAdj + i*numData*numData,numData,numData);
	}

	printf("\n\nMatch Test\n\n");

	makeFields_reg(
		regions,
		regions,
		keySize,
		numData - 1,
		numFields,
		stepSize,
		fieldSize,
		2,
		sigma,
		sigma*3,
		layer0, 
		layer1,
		-1,
		unitFieldsOut, 
		selectorFieldsOut);

	for (int i = 0; i < numFields; ++i){
		printf("\n\nFilter %d:\n", i);


		printf("\nUnit Weights %d:\n", i);
		printMatrix(unitFieldsOut + i*fieldSize*fieldSize,fieldSize,fieldSize);
		for (int j = 0; j < finalDim; ++j){
			printf("\nSelector Weights %d:\n", j);
			printMatrix(selectorFieldsOut + i*finalDim*fieldSize*fieldSize + j*fieldSize*fieldSize,fieldSize,fieldSize);
		}
	}

	printf("\n\nUnmatch Test\n\n");

	makeFields_reg(
		regions,
		regions + keySize,
		keySize,
		numData - 1,
		numFields,
		stepSize,
		fieldSize,
		2,
		sigma,
		sigma*3,
		layer0, 
		layer1,
		-1,
		unitFieldsOut, 
		selectorFieldsOut);

	for (int i = 0; i < numFields; ++i){
		printf("\n\nFilter %d:\n", i);


		printf("\nUnit Weights %d:\n", i);
		printMatrix(unitFieldsOut + i*fieldSize*fieldSize,fieldSize,fieldSize);
		for (int j = 0; j < finalDim; ++j){
			printf("\nSelector Weights %d:\n", j);
			printMatrix(selectorFieldsOut + i*finalDim*fieldSize*fieldSize + j*fieldSize*fieldSize,fieldSize,fieldSize);
		}
	}



/*
	batchMakeFields(
		queryLocsPointers,
		numTestPoints,
		numRegions,
		numLoc,
		stepSize,
		numFields,
		fieldSize,
		2,
		sigma,
		sigma,
		layer0, 
		layer1,
		unitFieldsOut, 
		selectorFieldsOut,
		2);
	for(int k = 0; k < numTestPoints; ++k){
		for (int i = 0; i < finalDim; ++i){
			printf("\n\nFilter %d:\n", i);


			printf("\nUnit Weights %d:\n", i);
			printMatrix(unitFieldsOut + k*numFields*fieldSize*fieldSize + i*fieldSize*fieldSize,fieldSize,fieldSize);
			for (int j = 0; j < finalDim; ++j){
				printf("\nSelector Weights %d:\n", i);
				printMatrix(selectorFieldsOut + k*numFields*fieldSize*fieldSize*finalDim + i*finalDim*fieldSize*fieldSize + j*fieldSize*fieldSize,fieldSize,fieldSize);
			}
		}
		
	}


	free(selectorFieldsOut);
	free(selectorAdj);
	free(unitFieldsOut);
	free(unitAdj);
	free(regions);

*/

	return 0;
}