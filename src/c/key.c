#include "key.h"


uint calcKeyLen(uint dataLen)
{
	uint keyLen = (dataLen/KEYDATASIZE);
	if(dataLen % KEYDATASIZE){
		keyLen++;
	}
	return keyLen;
}

int checkIndex(kint *key, uint i)
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
		if(x[i] > y[i])
			return -1;
		if (x[i] < y[i])
			return 1;
	}
	return 0;

}

// From here https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable

int numberOfOneBits(kint *x, uint keyLen)
{
	int numberOfOneBits = 0;
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

int numberOfDiff(kint *x, kint *y, uint keyLen)
{
	int numberOfOneBits = 0;
	kint zj;
	for(uint j = 0;j<keyLen;j++){
		zj = x[j] ^ y[j];
		zj = zj - ((zj >> 1) & 0x55555555);                    // reuse input as temporary
		zj = (zj & 0x33333333) + ((zj >> 2) & 0x33333333);     // temp
		numberOfOneBits += (((zj + (zj >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}
	return numberOfOneBits;
}

int checkKeyEmpty(kint *key,uint keyLen)
{
	uint i = 0;
	for(i = 0; i < keyLen; i++){
		if(key[i])
			return 1;
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
	if(checkIndex(key,index))
		key[index/KEYDATASIZE] -= (1 << (KEYDATASIZE-1-j));
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

void convertFromFloatToKey(float * raw, kint *key,uint dataLen)
{
	uint keyLen = calcKeyLen(dataLen);
	memset(key,0,sizeof(kint)*keyLen);
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

void convertFromIntToKey(int * raw, kint *key,uint dataLen)
{

	uint keyLen = calcKeyLen(dataLen);
	memset(key,0,sizeof(kint)*keyLen);
	uint i = 0,m=0;
	for(i=0;i<dataLen;i++){
		m = (1 << (KEYDATASIZE -(i % KEYDATASIZE) -1));
		key[i/KEYDATASIZE] = (key[i/KEYDATASIZE] & ~m) | (-(0<raw[i]) & m);
	}
}
