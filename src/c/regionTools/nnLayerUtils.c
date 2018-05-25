#include "nnLayerUtils.h"

nnLayer *allocateLayer(uint inDim, uint outDim)
{
	nnLayer * layer = malloc(sizeof(nnLayer));
	layer->A = malloc(inDim*outDim*sizeof(float));
	layer->b = malloc(outDim*sizeof(float));
	layer->inDim = inDim;
	layer->outDim = outDim;
	return layer;
}

nnLayer * createLayer(float * A, float *b, uint inDim, uint outDim)
{
	//Allocating A and b together
	nnLayer * layer = malloc(sizeof(nnLayer));
	layer->A = A;
	layer->b = b;
	layer->inDim = inDim;
	layer->outDim = outDim;
	return layer;
}

nnLayer * createCopyLayer(float * A, float *b, uint inDim, uint outDim)
{
	
	//Allocating A and b together
	nnLayer * layer = allocateLayer(inDim,outDim);
	memcpy(layer->A,A,inDim*outDim*sizeof(float));
	memcpy(layer->b,b,outDim*sizeof(float));	
	return layer;
}

float absf(float g)
{
	unsigned int *gg;
	gg=(unsigned int*)&g;
	*(gg)&=2147483647u;
	return g;
}

nnLayer * createCopyPosLayer(float * A, float *b, uint inDim, uint outDim)
{
	
	//Allocating A and b together
	nnLayer * layer = allocateLayer(inDim,outDim);
	for(uint i = 0; i < outDim; ++i){
		for(uint j = 0; j < inDim; ++j){
			layer->A[i*inDim + j] = absf(A[i*inDim + j]);
		}
		layer->b[i] = absf(b[i]);
	}
	return layer;
}

void freeLayer(nnLayer *layer)
{	
	//As we allocated A and b together there needs to only be 1 free operation, not 2
	if(layer){
		if(layer->A){
			free(layer->A);
		}
		if(layer->b){
			free(layer->b);
		}
		free(layer);
	}
}

void evalLayer(nnLayer *layer, float * input, float * output)
{
	uint inDim = layer->inDim;
	uint outDim = layer->outDim;
	cblas_scopy (outDim, layer->b, 1, output, 1);
	cblas_sgemv (CblasRowMajor, CblasNoTrans, outDim,inDim,1, layer->A, inDim, input, 1, 1, output, 1);
}

void printFloatArr(float *arr, uint length){
	uint i = 0;
	printf("{");
	if(length>0){
		for(i=0;i<length-1;i++){
			if(arr[i] == FLT_MAX){
				printf("---,");
			} else {
				printf("%.3f,",arr[i]);
			}
			
		}	
		if(arr[length-1] == FLT_MAX){
			printf("---,");
		} else {
			printf("%.3f",arr[length-1]);
		}
	}
	printf("}\n");
}

void printMatrix(float *arr, uint inDim, uint outDim){
	uint i = 0;
	for(i=0;i<outDim;i++){
		printFloatArr(arr + inDim*i,inDim);
	}
}

void printLayer(nnLayer * layer)
{
	printf("Weight Matrix:\n");
	printMatrix(layer->A,layer->inDim,layer->outDim);
	printf("Bias:\n");
	printFloatArr(layer->b,layer->outDim);
}
	
void makeLayerPositive(nnLayer * layer)
{
	for(uint i = 0; i < layer->outDim * layer->inDim; i++){
		if(layer->A[i] < 0){
			layer->A[i] *= -layer->A[i];
		}
	}
}

nnLayer * copyLayerPositive(nnLayer *layer, char transpose)
{
	if(transpose == 't'){
		nnLayer *pos = allocateLayer(layer->inDim, layer->outDim);
		for(uint i = 0; i < layer->outDim; ++i){
			for(uint j = 0; j < layer->inDim; ++j){
				pos->A[j*layer->outDim + i] = layer->A[i*layer->inDim + j];
			}
		}
		memcpy(pos->b,layer->b,layer->outDim*sizeof(float));
		return pos;
	} else {
		nnLayer *pos = createCopyLayer(layer->A,layer->b,layer->inDim, layer->outDim);
		makeLayerPositive(pos);
		return pos;
	}
}

void getSelectorWeights(
	kint *x, 
	kint *y, 
	nnLayer *positiveSelectorLayer,
	float *weights)
{	
	uint inDim = positiveSelectorLayer->inDim;
	uint outDim = positiveSelectorLayer->outDim;
	kint compKey;
	memset(weights,0,inDim*sizeof(float));
	for(uint i=0;i<inDim;i++){
		compKey = x[i/KEYDATASIZE] ^ y[i/KEYDATASIZE];
		if(compKey & (1 << (KEYDATASIZE-1-(i % KEYDATASIZE)))){
			cblas_saxpy(
				outDim,
				1.0,
				positiveSelectorLayer->A + i,
				inDim,
				weights,
				1);
			
		}
	}
}

void getSelectorWeightsT(
	kint *x, 
	kint *y, 
	nnLayer *positiveSelectorLayer,
	float *weights,
	uint offset)
{	
	uint inDim = positiveSelectorLayer->inDim;
	uint outDim = positiveSelectorLayer->outDim;
	kint compKey;
	for(uint i=0;i<outDim;i++){
		weights[i*offset] = 0;
	}

	for(uint i=0;i<inDim;i++){
		compKey = x[i/KEYDATASIZE] ^ y[i/KEYDATASIZE];
		if(compKey & (1 << (KEYDATASIZE-1-(i % KEYDATASIZE)))){

			cblas_saxpy(
				outDim,
				1.0,
				positiveSelectorLayer->A + i*outDim,
				1,
				weights,
				offset);
		}
	}
}

float getSelectorWeight(kint *x, kint *y,float * selectionVec, int dataLen)
{

	kint compKey;
	float selectorDist = 0.0;
	for(int i=0;i<dataLen;i++){
		compKey = x[i/KEYDATASIZE] ^ y[i/KEYDATASIZE];
		if(compKey & (1 << (KEYDATASIZE-1-(i % KEYDATASIZE)))){
			selectorDist += selectionVec[i];
		}
	}
	return selectorDist;
}

/*
union {
    uint32_t i[8];
    __m256   xmm;
} mask8[256] = {
 {{  0,  0,  0,  0,  0,  0,  0,  0 }}, {{  0,  0,  0,  0,  0,  0,  0, ~0 }}, {{  0,  0,  0,  0,  0,  0, ~0,  0 }}, {{  0,  0,  0,  0,  0,  0, ~0, ~0 }}, 
 {{  0,  0,  0,  0,  0, ~0,  0,  0 }}, {{  0,  0,  0,  0,  0, ~0,  0, ~0 }}, {{  0,  0,  0,  0,  0, ~0, ~0,  0 }}, {{  0,  0,  0,  0,  0, ~0, ~0, ~0 }}, 
 {{  0,  0,  0,  0, ~0,  0,  0,  0 }}, {{  0,  0,  0,  0, ~0,  0,  0, ~0 }}, {{  0,  0,  0,  0, ~0,  0, ~0,  0 }}, {{  0,  0,  0,  0, ~0,  0, ~0, ~0 }}, 
 {{  0,  0,  0,  0, ~0, ~0,  0,  0 }}, {{  0,  0,  0,  0, ~0, ~0,  0, ~0 }}, {{  0,  0,  0,  0, ~0, ~0, ~0,  0 }}, {{  0,  0,  0,  0, ~0, ~0, ~0, ~0 }},
 {{  0,  0,  0, ~0,  0,  0,  0,  0 }}, {{  0,  0,  0, ~0,  0,  0,  0, ~0 }}, {{  0,  0,  0, ~0,  0,  0, ~0,  0 }}, {{  0,  0,  0, ~0,  0,  0, ~0, ~0 }}, 
 {{  0,  0,  0, ~0,  0, ~0,  0,  0 }}, {{  0,  0,  0, ~0,  0, ~0,  0, ~0 }}, {{  0,  0,  0, ~0,  0, ~0, ~0,  0 }}, {{  0,  0,  0, ~0,  0, ~0, ~0, ~0 }}, 
 {{  0,  0,  0, ~0, ~0,  0,  0,  0 }}, {{  0,  0,  0, ~0, ~0,  0,  0, ~0 }}, {{  0,  0,  0, ~0, ~0,  0, ~0,  0 }}, {{  0,  0,  0, ~0, ~0,  0, ~0, ~0 }}, 
 {{  0,  0,  0, ~0, ~0, ~0,  0,  0 }}, {{  0,  0,  0, ~0, ~0, ~0,  0, ~0 }}, {{  0,  0,  0, ~0, ~0, ~0, ~0,  0 }}, {{  0,  0,  0, ~0, ~0, ~0, ~0, ~0 }}, 
 {{  0,  0, ~0,  0,  0,  0,  0,  0 }}, {{  0,  0, ~0,  0,  0,  0,  0, ~0 }}, {{  0,  0, ~0,  0,  0,  0, ~0,  0 }}, {{  0,  0, ~0,  0,  0,  0, ~0, ~0 }}, 
 {{  0,  0, ~0,  0,  0, ~0,  0,  0 }}, {{  0,  0, ~0,  0,  0, ~0,  0, ~0 }}, {{  0,  0, ~0,  0,  0, ~0, ~0,  0 }}, {{  0,  0, ~0,  0,  0, ~0, ~0, ~0 }}, 
 {{  0,  0, ~0,  0, ~0,  0,  0,  0 }}, {{  0,  0, ~0,  0, ~0,  0,  0, ~0 }}, {{  0,  0, ~0,  0, ~0,  0, ~0,  0 }}, {{  0,  0, ~0,  0, ~0,  0, ~0, ~0 }}, 
 {{  0,  0, ~0,  0, ~0, ~0,  0,  0 }}, {{  0,  0, ~0,  0, ~0, ~0,  0, ~0 }}, {{  0,  0, ~0,  0, ~0, ~0, ~0,  0 }}, {{  0,  0, ~0,  0, ~0, ~0, ~0, ~0 }}, 
 {{  0,  0, ~0, ~0,  0,  0,  0,  0 }}, {{  0,  0, ~0, ~0,  0,  0,  0, ~0 }}, {{  0,  0, ~0, ~0,  0,  0, ~0,  0 }}, {{  0,  0, ~0, ~0,  0,  0, ~0, ~0 }}, 
 {{  0,  0, ~0, ~0,  0, ~0,  0,  0 }}, {{  0,  0, ~0, ~0,  0, ~0,  0, ~0 }}, {{  0,  0, ~0, ~0,  0, ~0, ~0,  0 }}, {{  0,  0, ~0, ~0,  0, ~0, ~0, ~0 }}, 
 {{  0,  0, ~0, ~0, ~0,  0,  0,  0 }}, {{  0,  0, ~0, ~0, ~0,  0,  0, ~0 }}, {{  0,  0, ~0, ~0, ~0,  0, ~0,  0 }}, {{  0,  0, ~0, ~0, ~0,  0, ~0, ~0 }}, 
 {{  0,  0, ~0, ~0, ~0, ~0,  0,  0 }}, {{  0,  0, ~0, ~0, ~0, ~0,  0, ~0 }}, {{  0,  0, ~0, ~0, ~0, ~0, ~0,  0 }}, {{  0,  0, ~0, ~0, ~0, ~0, ~0, ~0 }},
 {{  0, ~0,  0,  0,  0,  0,  0,  0 }}, {{  0, ~0,  0,  0,  0,  0,  0, ~0 }}, {{  0, ~0,  0,  0,  0,  0, ~0,  0 }}, {{  0, ~0,  0,  0,  0,  0, ~0, ~0 }}, 
 {{  0, ~0,  0,  0,  0, ~0,  0,  0 }}, {{  0, ~0,  0,  0,  0, ~0,  0, ~0 }}, {{  0, ~0,  0,  0,  0, ~0, ~0,  0 }}, {{  0, ~0,  0,  0,  0, ~0, ~0, ~0 }}, 
 {{  0, ~0,  0,  0, ~0,  0,  0,  0 }}, {{  0, ~0,  0,  0, ~0,  0,  0, ~0 }}, {{  0, ~0,  0,  0, ~0,  0, ~0,  0 }}, {{  0, ~0,  0,  0, ~0,  0, ~0, ~0 }}, 
 {{  0, ~0,  0,  0, ~0, ~0,  0,  0 }}, {{  0, ~0,  0,  0, ~0, ~0,  0, ~0 }}, {{  0, ~0,  0,  0, ~0, ~0, ~0,  0 }}, {{  0, ~0,  0,  0, ~0, ~0, ~0, ~0 }},
 {{  0, ~0,  0, ~0,  0,  0,  0,  0 }}, {{  0, ~0,  0, ~0,  0,  0,  0, ~0 }}, {{  0, ~0,  0, ~0,  0,  0, ~0,  0 }}, {{  0, ~0,  0, ~0,  0,  0, ~0, ~0 }}, 
 {{  0, ~0,  0, ~0,  0, ~0,  0,  0 }}, {{  0, ~0,  0, ~0,  0, ~0,  0, ~0 }}, {{  0, ~0,  0, ~0,  0, ~0, ~0,  0 }}, {{  0, ~0,  0, ~0,  0, ~0, ~0, ~0 }}, 
 {{  0, ~0,  0, ~0, ~0,  0,  0,  0 }}, {{  0, ~0,  0, ~0, ~0,  0,  0, ~0 }}, {{  0, ~0,  0, ~0, ~0,  0, ~0,  0 }}, {{  0, ~0,  0, ~0, ~0,  0, ~0, ~0 }}, 
 {{  0, ~0,  0, ~0, ~0, ~0,  0,  0 }}, {{  0, ~0,  0, ~0, ~0, ~0,  0, ~0 }}, {{  0, ~0,  0, ~0, ~0, ~0, ~0,  0 }}, {{  0, ~0,  0, ~0, ~0, ~0, ~0, ~0 }}, 
 {{  0, ~0, ~0,  0,  0,  0,  0,  0 }}, {{  0, ~0, ~0,  0,  0,  0,  0, ~0 }}, {{  0, ~0, ~0,  0,  0,  0, ~0,  0 }}, {{  0, ~0, ~0,  0,  0,  0, ~0, ~0 }}, 
 {{  0, ~0, ~0,  0,  0, ~0,  0,  0 }}, {{  0, ~0, ~0,  0,  0, ~0,  0, ~0 }}, {{  0, ~0, ~0,  0,  0, ~0, ~0,  0 }}, {{  0, ~0, ~0,  0,  0, ~0, ~0, ~0 }}, 
 {{  0, ~0, ~0,  0, ~0,  0,  0,  0 }}, {{  0, ~0, ~0,  0, ~0,  0,  0, ~0 }}, {{  0, ~0, ~0,  0, ~0,  0, ~0,  0 }}, {{  0, ~0, ~0,  0, ~0,  0, ~0, ~0 }}, 
 {{  0, ~0, ~0,  0, ~0, ~0,  0,  0 }}, {{  0, ~0, ~0,  0, ~0, ~0,  0, ~0 }}, {{  0, ~0, ~0,  0, ~0, ~0, ~0,  0 }}, {{  0, ~0, ~0,  0, ~0, ~0, ~0, ~0 }}, 
 {{  0, ~0, ~0, ~0,  0,  0,  0,  0 }}, {{  0, ~0, ~0, ~0,  0,  0,  0, ~0 }}, {{  0, ~0, ~0, ~0,  0,  0, ~0,  0 }}, {{  0, ~0, ~0, ~0,  0,  0, ~0, ~0 }}, 
 {{  0, ~0, ~0, ~0,  0, ~0,  0,  0 }}, {{  0, ~0, ~0, ~0,  0, ~0,  0, ~0 }}, {{  0, ~0, ~0, ~0,  0, ~0, ~0,  0 }}, {{  0, ~0, ~0, ~0,  0, ~0, ~0, ~0 }}, 
 {{  0, ~0, ~0, ~0, ~0,  0,  0,  0 }}, {{  0, ~0, ~0, ~0, ~0,  0,  0, ~0 }}, {{  0, ~0, ~0, ~0, ~0,  0, ~0,  0 }}, {{  0, ~0, ~0, ~0, ~0,  0, ~0, ~0 }}, 
 {{  0, ~0, ~0, ~0, ~0, ~0,  0,  0 }}, {{  0, ~0, ~0, ~0, ~0, ~0,  0, ~0 }}, {{  0, ~0, ~0, ~0, ~0, ~0, ~0,  0 }}, {{  0, ~0, ~0, ~0, ~0, ~0, ~0, ~0 }},
 {{ ~0,  0,  0,  0,  0,  0,  0,  0 }}, {{ ~0,  0,  0,  0,  0,  0,  0, ~0 }}, {{ ~0,  0,  0,  0,  0,  0, ~0,  0 }}, {{ ~0,  0,  0,  0,  0,  0, ~0, ~0 }}, 
 {{ ~0,  0,  0,  0,  0, ~0,  0,  0 }}, {{ ~0,  0,  0,  0,  0, ~0,  0, ~0 }}, {{ ~0,  0,  0,  0,  0, ~0, ~0,  0 }}, {{ ~0,  0,  0,  0,  0, ~0, ~0, ~0 }}, 
 {{ ~0,  0,  0,  0, ~0,  0,  0,  0 }}, {{ ~0,  0,  0,  0, ~0,  0,  0, ~0 }}, {{ ~0,  0,  0,  0, ~0,  0, ~0,  0 }}, {{ ~0,  0,  0,  0, ~0,  0, ~0, ~0 }}, 
 {{ ~0,  0,  0,  0, ~0, ~0,  0,  0 }}, {{ ~0,  0,  0,  0, ~0, ~0,  0, ~0 }}, {{ ~0,  0,  0,  0, ~0, ~0, ~0,  0 }}, {{ ~0,  0,  0,  0, ~0, ~0, ~0, ~0 }},
 {{ ~0,  0,  0, ~0,  0,  0,  0,  0 }}, {{ ~0,  0,  0, ~0,  0,  0,  0, ~0 }}, {{ ~0,  0,  0, ~0,  0,  0, ~0,  0 }}, {{ ~0,  0,  0, ~0,  0,  0, ~0, ~0 }}, 
 {{ ~0,  0,  0, ~0,  0, ~0,  0,  0 }}, {{ ~0,  0,  0, ~0,  0, ~0,  0, ~0 }}, {{ ~0,  0,  0, ~0,  0, ~0, ~0,  0 }}, {{ ~0,  0,  0, ~0,  0, ~0, ~0, ~0 }}, 
 {{ ~0,  0,  0, ~0, ~0,  0,  0,  0 }}, {{ ~0,  0,  0, ~0, ~0,  0,  0, ~0 }}, {{ ~0,  0,  0, ~0, ~0,  0, ~0,  0 }}, {{ ~0,  0,  0, ~0, ~0,  0, ~0, ~0 }}, 
 {{ ~0,  0,  0, ~0, ~0, ~0,  0,  0 }}, {{ ~0,  0,  0, ~0, ~0, ~0,  0, ~0 }}, {{ ~0,  0,  0, ~0, ~0, ~0, ~0,  0 }}, {{ ~0,  0,  0, ~0, ~0, ~0, ~0, ~0 }}, 
 {{ ~0,  0, ~0,  0,  0,  0,  0,  0 }}, {{ ~0,  0, ~0,  0,  0,  0,  0, ~0 }}, {{ ~0,  0, ~0,  0,  0,  0, ~0,  0 }}, {{ ~0,  0, ~0,  0,  0,  0, ~0, ~0 }}, 
 {{ ~0,  0, ~0,  0,  0, ~0,  0,  0 }}, {{ ~0,  0, ~0,  0,  0, ~0,  0, ~0 }}, {{ ~0,  0, ~0,  0,  0, ~0, ~0,  0 }}, {{ ~0,  0, ~0,  0,  0, ~0, ~0, ~0 }}, 
 {{ ~0,  0, ~0,  0, ~0,  0,  0,  0 }}, {{ ~0,  0, ~0,  0, ~0,  0,  0, ~0 }}, {{ ~0,  0, ~0,  0, ~0,  0, ~0,  0 }}, {{ ~0,  0, ~0,  0, ~0,  0, ~0, ~0 }}, 
 {{ ~0,  0, ~0,  0, ~0, ~0,  0,  0 }}, {{ ~0,  0, ~0,  0, ~0, ~0,  0, ~0 }}, {{ ~0,  0, ~0,  0, ~0, ~0, ~0,  0 }}, {{ ~0,  0, ~0,  0, ~0, ~0, ~0, ~0 }}, 
 {{ ~0,  0, ~0, ~0,  0,  0,  0,  0 }}, {{ ~0,  0, ~0, ~0,  0,  0,  0, ~0 }}, {{ ~0,  0, ~0, ~0,  0,  0, ~0,  0 }}, {{ ~0,  0, ~0, ~0,  0,  0, ~0, ~0 }}, 
 {{ ~0,  0, ~0, ~0,  0, ~0,  0,  0 }}, {{ ~0,  0, ~0, ~0,  0, ~0,  0, ~0 }}, {{ ~0,  0, ~0, ~0,  0, ~0, ~0,  0 }}, {{ ~0,  0, ~0, ~0,  0, ~0, ~0, ~0 }}, 
 {{ ~0,  0, ~0, ~0, ~0,  0,  0,  0 }}, {{ ~0,  0, ~0, ~0, ~0,  0,  0, ~0 }}, {{ ~0,  0, ~0, ~0, ~0,  0, ~0,  0 }}, {{ ~0,  0, ~0, ~0, ~0,  0, ~0, ~0 }}, 
 {{ ~0,  0, ~0, ~0, ~0, ~0,  0,  0 }}, {{ ~0,  0, ~0, ~0, ~0, ~0,  0, ~0 }}, {{ ~0,  0, ~0, ~0, ~0, ~0, ~0,  0 }}, {{ ~0,  0, ~0, ~0, ~0, ~0, ~0, ~0 }},
 {{ ~0, ~0,  0,  0,  0,  0,  0,  0 }}, {{ ~0, ~0,  0,  0,  0,  0,  0, ~0 }}, {{ ~0, ~0,  0,  0,  0,  0, ~0,  0 }}, {{ ~0, ~0,  0,  0,  0,  0, ~0, ~0 }}, 
 {{ ~0, ~0,  0,  0,  0, ~0,  0,  0 }}, {{ ~0, ~0,  0,  0,  0, ~0,  0, ~0 }}, {{ ~0, ~0,  0,  0,  0, ~0, ~0,  0 }}, {{ ~0, ~0,  0,  0,  0, ~0, ~0, ~0 }}, 
 {{ ~0, ~0,  0,  0, ~0,  0,  0,  0 }}, {{ ~0, ~0,  0,  0, ~0,  0,  0, ~0 }}, {{ ~0, ~0,  0,  0, ~0,  0, ~0,  0 }}, {{ ~0, ~0,  0,  0, ~0,  0, ~0, ~0 }}, 
 {{ ~0, ~0,  0,  0, ~0, ~0,  0,  0 }}, {{ ~0, ~0,  0,  0, ~0, ~0,  0, ~0 }}, {{ ~0, ~0,  0,  0, ~0, ~0, ~0,  0 }}, {{ ~0, ~0,  0,  0, ~0, ~0, ~0, ~0 }},
 {{ ~0, ~0,  0, ~0,  0,  0,  0,  0 }}, {{ ~0, ~0,  0, ~0,  0,  0,  0, ~0 }}, {{ ~0, ~0,  0, ~0,  0,  0, ~0,  0 }}, {{ ~0, ~0,  0, ~0,  0,  0, ~0, ~0 }}, 
 {{ ~0, ~0,  0, ~0,  0, ~0,  0,  0 }}, {{ ~0, ~0,  0, ~0,  0, ~0,  0, ~0 }}, {{ ~0, ~0,  0, ~0,  0, ~0, ~0,  0 }}, {{ ~0, ~0,  0, ~0,  0, ~0, ~0, ~0 }}, 
 {{ ~0, ~0,  0, ~0, ~0,  0,  0,  0 }}, {{ ~0, ~0,  0, ~0, ~0,  0,  0, ~0 }}, {{ ~0, ~0,  0, ~0, ~0,  0, ~0,  0 }}, {{ ~0, ~0,  0, ~0, ~0,  0, ~0, ~0 }}, 
 {{ ~0, ~0,  0, ~0, ~0, ~0,  0,  0 }}, {{ ~0, ~0,  0, ~0, ~0, ~0,  0, ~0 }}, {{ ~0, ~0,  0, ~0, ~0, ~0, ~0,  0 }}, {{ ~0, ~0,  0, ~0, ~0, ~0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0,  0,  0,  0,  0,  0 }}, {{ ~0, ~0, ~0,  0,  0,  0,  0, ~0 }}, {{ ~0, ~0, ~0,  0,  0,  0, ~0,  0 }}, {{ ~0, ~0, ~0,  0,  0,  0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0,  0,  0, ~0,  0,  0 }}, {{ ~0, ~0, ~0,  0,  0, ~0,  0, ~0 }}, {{ ~0, ~0, ~0,  0,  0, ~0, ~0,  0 }}, {{ ~0, ~0, ~0,  0,  0, ~0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0,  0, ~0,  0,  0,  0 }}, {{ ~0, ~0, ~0,  0, ~0,  0,  0, ~0 }}, {{ ~0, ~0, ~0,  0, ~0,  0, ~0,  0 }}, {{ ~0, ~0, ~0,  0, ~0,  0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0,  0, ~0, ~0,  0,  0 }}, {{ ~0, ~0, ~0,  0, ~0, ~0,  0, ~0 }}, {{ ~0, ~0, ~0,  0, ~0, ~0, ~0,  0 }}, {{ ~0, ~0, ~0,  0, ~0, ~0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0, ~0,  0,  0,  0,  0 }}, {{ ~0, ~0, ~0, ~0,  0,  0,  0, ~0 }}, {{ ~0, ~0, ~0, ~0,  0,  0, ~0,  0 }}, {{ ~0, ~0, ~0, ~0,  0,  0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0, ~0,  0, ~0,  0,  0 }}, {{ ~0, ~0, ~0, ~0,  0, ~0,  0, ~0 }}, {{ ~0, ~0, ~0, ~0,  0, ~0, ~0,  0 }}, {{ ~0, ~0, ~0, ~0,  0, ~0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0, ~0, ~0,  0,  0,  0 }}, {{ ~0, ~0, ~0, ~0, ~0,  0,  0, ~0 }}, {{ ~0, ~0, ~0, ~0, ~0,  0, ~0,  0 }}, {{ ~0, ~0, ~0, ~0, ~0,  0, ~0, ~0 }}, 
 {{ ~0, ~0, ~0, ~0, ~0, ~0,  0,  0 }}, {{ ~0, ~0, ~0, ~0, ~0, ~0,  0, ~0 }}, {{ ~0, ~0, ~0, ~0, ~0, ~0, ~0,  0 }}, {{ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0 }}  
};
float getSelectorWeight(kint *x, kint *y,float * selectionVec, int numFloat)
{   
    float sum = 0;
    __m256 sum8 = { 0.0 };
    __m256* a8;
    int i = 0;
    kint b = 0;
    for(; i + KEYDATASIZE < numFloat; i += KEYDATASIZE){
        b = x[i/KEYDATASIZE] ^ y[i/KEYDATASIZE];
        printf("X:  %u\n", x[i/KEYDATASIZE]);
        printf("Y:  %u\n", y[i/KEYDATASIZE]);
        a8 = (__m256*)(selectionVec + i);
        for(int j = i; j < i+KEYDATASIZE; j++){
        	printf("A[%d]: %f  ", j, selectionVec[j]);
        }
        printf("\n\n");
        for (int j = 0; j < 4; j++, b>>=8){
	        printf("a8: %f %f %f %f %f %f %f %f\n", a8[3-j][0], a8[3-j][1], a8[3-j][2], a8[3-j][3], a8[3-j][4], a8[3-j][5], a8[3-j][6], a8[3-j][7]);
            sum8 += _mm256_and_ps(a8[3-j], mask8[b&0xFF].xmm);
        }
    }
    sum = sum8[0] + sum8[1] + sum8[2] + sum8[3] + sum8[4] + sum8[5] + sum8[6] + sum8[7];
    for(;i<numFloat;++i){
        b = x[i/KEYDATASIZE] ^ y[i/KEYDATASIZE];
        if(b & (1 << (KEYDATASIZE-1-(i % KEYDATASIZE)))){
            sum += selectionVec[i];
        }
    }
    return sum;
}
*/
/*
float getSelectorWeightsSpread(kint *x, kint *y,float * selectionVec, int numHP, int numSel, int numFields, float * output, int i, int j)
{   
    float sum = 0;
    __m256 sum8 = { 0.0 };
    __m256 masked = { 0.0 };
    __m256* a8;
    int i = 0;
    kint b = 0;
    for(; i + KEYDATASIZE < numFloat; i += KEYDATASIZE){
        b = x[i/KEYDATASIZE] ^ y[i/KEYDATASIZE];
        a8 = (__m256*)(selectionVec + i);
        for (int j = 0; j < 4; j++, b>>=8){
        	masked = mask8[b&0xFF].xmm;
            sum8 += _mm256_and_ps(a8[3-j], masked);
        }
    }
    sum = sum8[0] + sum8[1] + sum8[2] + sum8[3] + sum8[4] + sum8[5] + sum8[6] + sum8[7];
    for(;i<numFloat;++i){
        if(bitMask[i/KEYDATASIZE] & (1 << (KEYDATASIZE-1-(i % KEYDATASIZE)))){
            sum += selectionVec[i];
        }
    }
    return sum;
}
*/