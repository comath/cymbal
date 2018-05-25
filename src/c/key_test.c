#include <stdio.h>
#include <stdlib.h>
#include "key.h"

int compareArr(int * x, int * y, int length)
{
	int i = 0;
	for(i = 0; i < length; i++){
		if(x[i]>y[i]){
			return -1;
		} 
		if (x[i]<y[i]){
			return 1;
		}
	}
	return 0;
}

int * createRandomBinaryArray(int numElements)
{
	
	int * randArr = malloc(numElements*sizeof(int));
	int i = 0;

	for(i=0;i<numElements;i++){
		randArr[i] = (rand() % 2);
	}
	return randArr;
}

void randomizePoint(float *p, uint dim)
{
	uint i = 0;
	for(i=0;i<dim;i++){
		p[i] = (double)((rand() % 20000) - 10000)/1000;
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

void fillRandomBinaryArray(int * arr, int numElements)
{
	int i = 0;
	for(i=0;i<numElements;i++){
		arr[i] = (rand() % 2);
	}
}

int countNumOnes(int* arr, int numElements)
{
	int numOnes = 0;
	for (int i = 0; i < numElements; ++i){
		if(arr[i] == 1){
			numOnes++;
		}
	}
	return numOnes;
}

int countDiff(int* arr1, int* arr2, int numElements)
{
	int diff = 0;
	for (int i = 0; i < numElements; ++i){
		if(arr1[i] != arr2[i]){
			diff++;
		}
	}
	return diff;
}

void printCharArr(char * arr, int numElements){
	int i = 0;
	printf("[");
	for(i=0;i<numElements-1;i++){
		printf("%u,",arr[i]);
	}
	printf("%u", arr[numElements-1]);
	printf("]\n");
}

void printKeyArr(kint *key, uint length){
	uint i = 0;
	printf("[");
	for(i=0;i<length-1;i++){
		printf("%u,",key[i]);
	}
	printf("%u", key[length-1]);
	printf("]\n");
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



void testCompareKey()
{
	
	uint arrLength = 100;
	int *arr = createRandomBinaryArray(arrLength);
	int *arr2 = createRandomBinaryArray(arrLength);
	uint keyLen = calcKeyLen(arrLength);
	//printf("Key length: %u\n", keyLen);

	kint *testKey = malloc(keyLen*sizeof(kint));
	kint *testKey2 = malloc(keyLen*sizeof(kint));
	convertFromIntToKey(arr,testKey,arrLength);
	convertFromIntToKey(arr2,testKey2,arrLength);
	if( (compareKey( testKey, testKey2,keyLen)  != compareArr(arr,arr2,arrLength)) ) {
		printf("Failure, Test Func: %d Actual Val: %d\n",compareArr(arr,arr2,arrLength),compareKey( testKey, testKey2,keyLen));
	} 
	free(arr);
	free(arr2);
	free(testKey);
	free(testKey2);
}

void testIntKey()
{
	
	int arrLength = 100;
	int *arr = createRandomBinaryArray(arrLength);
	int *RecreateArr = malloc(arrLength*sizeof(int));
	
	uint keyLen = calcKeyLen(arrLength);
	kint *testKey = malloc(keyLen*sizeof(kint));

	convertFromIntToKey(arr,testKey,arrLength);
	convertFromKeyToInt(testKey,RecreateArr,arrLength);
	if(compareArr(arr,RecreateArr,arrLength) != 0){
		printf("Faliure, Recreated Arr:\n");
		printIntArr(RecreateArr,arrLength);
		printf("Original one:\n");
		printIntArr(arr,arrLength);
	}


	free(arr);
	free(RecreateArr);
	free(testKey);
}

void testFloatRebuild()
{
	int arrLength = 100;
	float *arr = malloc(arrLength*sizeof(float));
	randomizePoint(arr,arrLength);
	float *RecreateArr = malloc(arrLength*sizeof(float));
	
	uint keyLen = calcKeyLen(arrLength);
	kint *testKey = malloc(keyLen*sizeof(kint));

	convertFromFloatToKey(arr,testKey,arrLength);
	convertFromKeyToFloat(testKey,RecreateArr,arrLength);

	for(int i = 0; i < arrLength; ++i){
		if(arr[i] < 0 && checkIndex(testKey, i)){
			printf("Failure on %d\n", i);
		}
	}
}

void testOffByN()
{
	uint arrLength = 100;
	int *arr = createRandomBinaryArray(arrLength);
	int *arr2 = createRandomBinaryArray(arrLength);
	uint keyLen = calcKeyLen(arrLength);
	//printf("Key length: %u\n", keyLen);

	kint *testKey = malloc(keyLen*sizeof(kint));
	kint *testKey2 = malloc(keyLen*sizeof(kint));
	convertFromIntToKey(arr,testKey,arrLength);
	convertFromIntToKey(arr2,testKey2,arrLength);

	if(numberOfOneBits(testKey, keyLen) != countNumOnes(arr, arrLength)){
		printf("numberOfOneBits Failed");
		printIntArr(arr,arrLength);
	}

	if(numberOfDiff(testKey, testKey2, keyLen) != countDiff(arr, arr2, arrLength)){
		printf("numberDiff Failed");
		printIntArr(arr,arrLength);
	}
	free(arr);
	free(arr2);
	free(testKey);
	free(testKey2);
}

int main()
{
	srand(time(NULL));
	int i = 0;
	printf("If no faliures are printed then we are fine.\n");
	printf("testCompareKey:\n");
	testCompareKey();
	for(i=0;i<10;i++){	testCompareKey(); }
	printf("testIntKey:\n");
	for(i=0;i<10;i++){	testRebuildKey(); }
	printf("testoffByN:\n");
	for(i=0;i<10;i++){	testOffByN(); }
	printf("testFloatRebuild:\n");
	for(i=0;i<10;i++){	testFloatRebuild(); }

	return 0;
}