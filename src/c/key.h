#ifndef _key_h
#define _key_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
/*
We store the sets bit packed into unsigned 32 bit integers. This makes all future operations faster and simpler to write out
*/
#define kint uint32_t  //Definition of kint, makes it easier to resize later if needed
#define KEYDATASIZE 32

void printKey(kint *key, uint dataLen); // Prints the set associated to the key. If dataLen is unknown at time of call, just call with 32*keyLen
uint calcKeyLen(uint dataLen); // Returns the length of key needed to store a 

// Functions to handle key interactions
int compareKey(kint *x, kint *y, uint keyLength); // Lexographically compares two keys. See keyTest for the required behavior.
/*
Modify individual elements of the bit packed sets.
*/
void addIndexToKey(kint * key, uint index); // Changes the bit at index i from 0 to 1. Does nothing if bit is already 1
void removeIndexFromKey(kint * key, uint index); // Changes the bit at index i from 1 to 0. Does nothing if bit is already 0

int checkIndex(kint * key, uint index); // Returns a positive integer if the bit at the provided index is set
int checkKeyEmpty(kint *key, uint keyLength); // Returns 1 if the key is empty

/* 
Counts the cardinality of a set, and the cardinality of the set difference.
*/
int numberOfOneBits(kint *x, uint keyLength); 
int numberOfDiff(kint *x, kint *y, uint keyLength); 

/*

*/
void convertFromIntToKey(int * raw, kint *key,uint dataLen);
void convertFromKeyToInt(kint *key, int * output, uint dataLen);

void convertFromFloatToKey(float * raw, kint *key,uint dataLen);
void convertFromKeyToFloat(kint *key, float * output, uint dataLen);


#endif