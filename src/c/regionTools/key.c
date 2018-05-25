#include "key.h"


uint calcKeyLen(uint dataLen)
{
	uint keyLen = (dataLen/KEYDATASIZE);
	if(dataLen % KEYDATASIZE){
		keyLen++;
	}
	return keyLen;
}

uint checkIndex(kint *key, uint i)
{
	if(key[i/KEYDATASIZE] & (1 << (KEYDATASIZE-1-(i % KEYDATASIZE)))){
		return 1;
	} else {
		return 0;
	}
}


int compareKey(kint *x, kint *y, uint keyLen)
{
	//return memcmp (x,y, keyLen*sizeof(kint));

	uint i = 0;
	
	for(i = 0; i < keyLen; i++){
		if(x[i] > y[i]){
			return -1;
		} 
		if (x[i] < y[i]){
			return 1;
		}
	}
	return 0;

}

#define LEFTCHILD(X)  2*(X)+1
#define PARENT(X)  ((X) - 1)/2

void swapKeys(kint *x, kint *y, uint keyLen)
{
	kint swapSpace;
	for(uint i = 0; i < keyLen; ++i){
		swapSpace = x[i];
		x[i] = y[i];
		y[i] = swapSpace;
	}
}

void siftDownKeys(uint start, uint end, kint *keys, uint keyLen)
{
	uint root = start;
	uint swap = root;
	uint child = root;
	while(LEFTCHILD(root) < end){
		child = LEFTCHILD(root);
		swap = root;
		if( compareKey(keys + swap*keyLen,keys + child*keyLen,keyLen) < 0)
			swap = child;
		if(child + 1 < end && compareKey(keys + swap*keyLen,keys + (child + 1)*keyLen,keyLen) < 0)
			swap = child + 1;
		if(swap == root)
			return;
		else {
			swapKeys(keys + swap*keyLen,keys + root*keyLen,keyLen);
			root = swap;
		}

	}
}

void siftUpKeys(uint start, uint end, kint *keys, uint keyLen)
{
	uint child = end;
	uint parent = child;
	while(child > start){
		parent = PARENT(child);
		if(compareKey(keys + parent*keyLen,keys + child*keyLen,keyLen) < 0){
			swapKeys(keys + parent*keyLen,keys + child*keyLen,keyLen);

			child = parent;
		} else {
			return;
		}
	}
}

int checkMinHeap(uint start, uint end, kint *keys, uint keyLen )
{
	uint l_child = LEFTCHILD(start);
	uint r_child = LEFTCHILD(start) + 1;
	//printf("     [%u] Parent: %u\n",start,keys[start * keyLen]);
	if(	l_child < end){
		//printf("     [%u] L Child: %u, comp: %d\n",l_child,keys[l_child * keyLen], compareKey(keys + start*keyLen,keys + l_child*keyLen,keyLen));
		if(compareKey(keys + start*keyLen,keys + l_child*keyLen,keyLen) <= 0 ||
		checkMinHeap(l_child, end, keys, keyLen) == 0){
			return 0;
		} 
	} 
	if( r_child < end){
		//printf("     [%u] R Child: %u, comp: %d\n",r_child, keys[r_child * keyLen], compareKey(keys + start*keyLen,keys + r_child*keyLen,keyLen));
		if(	compareKey(keys + start*keyLen,keys + r_child*keyLen,keyLen) <= 0 ||
			checkMinHeap(r_child, end, keys, keyLen) == 0){
			return 0;
		}
	}
	return 1; 	
}

void orderKeys(kint * keys, uint keyLen, uint numKeys)
{
	// Build the heap
	for(uint i = 0; i < numKeys; ++i){
		//swapKeys(keys,keys + i*keyLen,keyLen);
		siftUpKeys(0,i,keys,keyLen);

	}

	// Heap Sort!
	for(uint i = numKeys - 1; i > 0; --i){
		swapKeys(keys,keys + i*keyLen,keyLen);
		
		siftDownKeys(0, i - 1, keys,keyLen);
		/*
		for(uint j = 0; j < i; ++j)
			printf("[%u] %u\n", j, keys[j]);
		printf("Sorted %u Heap %d\n",i, checkMinHeap(0,i-1,keys,keyLen));
		for(uint j = i; j < numKeys; ++j)
			printf("[%u] %u\n", j, keys[j]);
		*/
	}
}

int isPowerOfTwo (kint x)
{
  return ((x != 0) && !(x & (x - 1)));
}

int offByOne(kint *x, kint *y, uint keyLen)
{
	kint cmp = 0;
	uint i = 0;
	uint j = 0;
	// Find the first non-zero difference
	while(i<keyLen && cmp == 0){
		cmp = x[i] - y[i];
		i++; 
	}
	// check that it's the only non-zero difference 
	for(j = i; j<keyLen;j++){
		if(x[j] - y[j]){
			return 0;
		}
	}
	// cmp should contain the only non-zero entry of the difference
	// Checking that it's a power of two
	return isPowerOfTwo(cmp);
}

// From here https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable

unsigned int numberOfOneBits(kint *x, uint keyLen)
{
	unsigned int numberOfOneBits = 0;
	kint xi;
	uint i = 0;
	for(i = 0; i<keyLen;i++){
		xi = x[i]; // Copy x[i]
		xi = xi - ((xi >> 1) & 0x55555555);                    // reuse input as temporary
		xi = (xi & 0x33333333) + ((xi >> 2) & 0x33333333);     // temp
		numberOfOneBits += (((xi + (xi >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}
	return numberOfOneBits; 
}

unsigned int numberOfDiff(kint *x, kint *y, uint keyLen)
{
	unsigned int numberOfOneBits = 0;
	kint zj;
	for(uint j = 0;j<keyLen;j++){
		zj = x[j] ^ y[j];
		zj = zj - ((zj >> 1) & 0x55555555);                    // reuse input as temporary
		zj = (zj & 0x33333333) + ((zj >> 2) & 0x33333333);     // temp
		numberOfOneBits += (((zj + (zj >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}
	return numberOfOneBits;
}


int evalSig(kint *key, float *selectionVec, float selectionBias, uint dataLen)
{
	uint i = 0;
	float result = selectionBias;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			result += selectionVec[i];
		} 
	}
	if(result > 0){
		return 1;
	} else if(result < 0){
		return -1;
	} else {
		return 0;
	}
}

int checkEmptyKey(kint *key,uint keyLen)
{
	uint i = 0;
	for(i = 0; i < keyLen; i++){
		if(key[i]){
			return 1;
		}
	}
	return 0;
}

void addIndexToKey(kint *key, uint index)
{
	key[index/KEYDATASIZE] |= (1 << (KEYDATASIZE-1-(index % KEYDATASIZE)));
}

void removeIndexFromKey(kint *key, uint index)
{
	int j = index % KEYDATASIZE;
	if(checkIndex(key,index)){
		key[index/KEYDATASIZE] -= (1 << (KEYDATASIZE-1-j));
	}
}

void clearKey(kint *key, uint keyLen)
{
	memset(key,0,keyLen*sizeof(kint));
}

void printKey(kint *key, uint dataLen){
	uint i=0;
	printf("[");
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			printf("%u,",i);
		}
	}
	printf("]\n");
}



void copyKey(kint *key1, kint *key2, uint keyLen)
{
	memcpy(key2, key1, keyLen*sizeof(kint));
}


void chromaticKey(kint* key, float *rgb, uint dataLen)
{
	rgb[0] = 0;
	rgb[1] = 0;
	rgb[2] = 0;
	uint i = 0;
	for(i =0;i<dataLen;i++){
		if(checkIndex(key,i)){
			if(i % 3 == 0){ rgb[0]+= 1.0f/ (1 << (int)(i/3+1)); }
			if(i % 3 == 1){ rgb[1]+= 1.0f/ (1 << (int)(i/3+1)); }
			if(i % 3 == 2){ rgb[2]+= 1.0f/ (1 << (int)(i/3+1)); }
		}		
	}
}

void convertFromFloatToKey(float * raw, kint *key,uint dataLen)
{
	uint keyLen = calcKeyLen(dataLen);
	clearKey(key, keyLen);
	uint i = 0,m=0;
	for(i=0;i<dataLen;i++){
		m = (1 << (KEYDATASIZE -(i % KEYDATASIZE) -1));
		key[i/KEYDATASIZE] = (key[i/KEYDATASIZE] & ~m) | (-(0<raw[i]) & m);
	}
}

void convertFromKeyToFloat(kint *key, float * raw, uint dataLen)
{
	uint i = 0;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			raw[i] = 1.0;
		} else {
			raw[i] = 0.0;
		}
	}
}



void batchConvertFromIntToKey(int * raw, kint *key,uint dataLen, uint numData){
	uint i =0;
	uint keyLen = calcKeyLen(dataLen);
	for(i=0;i<numData;i++){
		convertFromIntToKey(raw + i*dataLen, key +i*keyLen,dataLen);
	}
}

void convertFromKeyToInt(kint *key, int * raw, uint dataLen)
{
	uint i = 0;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			raw[i] = 1;
		} else {
			raw[i] = 0;
		}
	}
}

void convertFromKeyToChar(kint *key, char * raw, uint dataLen)
{
	uint i = 0;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			raw[i] = 1;
		} else {
			raw[i] = 0;
		}
	}
}

void convertFromIntToKey(int * raw, kint *key,uint dataLen)
{

	uint keyLen = calcKeyLen(dataLen);
	clearKey(key, keyLen);
	uint i = 0,m=0;
	for(i=0;i<dataLen;i++){
		m = (1 << (KEYDATASIZE -(i % KEYDATASIZE) -1));
		key[i/KEYDATASIZE] = (key[i/KEYDATASIZE] & ~m) | (-(0<raw[i]) & m);
	}
}
