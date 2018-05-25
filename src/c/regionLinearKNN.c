#include "regionLinearKNN.h"

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

void swapPairs(int i, int j, regDistPair *knn)
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

void siftDown(int start, int end, regDistPair *knn)
{
	int root = start;
	int swap = root;
	int child = root;
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

void siftUp(int start, int end, regDistPair *knn)
{
	int child = end;
	int parent = child;
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

int checkMinHeap(int start, int end, regDistPair *keys)
{
	int l_child = LEFTCHILD(start);
	int r_child = LEFTCHILD(start) + 1;
	//printf("     [%u] Parent: %u\n",start,keys[start * keyLen]);
	if(	l_child < end){
		//printf("     [%u] L Child: %u, comp: %d\n",l_child,keys[l_child * keyLen], compareKey(keys + start*keyLen,keys + l_child*keyLen,keyLen));
		if(keys[start].dist < keys[l_child].dist ||
		checkMinHeap(l_child, end, keys) == 0){
			return 0;
		} 
	} 
	if( r_child < end){
		//printf("     [%u] R Child: %u, comp: %d\n",r_child, keys[r_child * keyLen], compareKey(keys + start*keyLen,keys + r_child*keyLen,keyLen));
		if(	keys[start].dist < keys[r_child].dist ||
			checkMinHeap(r_child, end, keys) == 0){
			return 0;
		}
	}
	return 1; 	
}


void regKNN(kint *queryKey, int keyLen, kint * regArr, int numLoc, int k, regDistPair *knn)
{
	int curDist = -1;
	// Build the initial heap
	for(int i = 0; i < k; ++i){
		knn[i].dist = numberOfDiff(queryKey,regArr + i*keyLen, keyLen);
		knn[i].regIndex = i;
		knn[i].reg = regArr + i*keyLen;
		siftUp(0,i,knn);
	}

	// We have a max heap
	for(int i = k; i < numLoc; ++i){
		curDist = numberOfDiff(queryKey,regArr + i*keyLen, keyLen);
		if(curDist < knn[0].dist){
			knn[0].dist = curDist;
			knn[0].regIndex = i;
			knn[0].reg = regArr + i*keyLen;
			siftDown(0, k - 1, knn);
		}
	}

	// Heap Sort!
	for(int i = k - 1; i > 0; --i){
		swapPairs(0,i,knn);
		siftDown(0, i - 1, knn);
	}
}

void localRegKNN(kint *queryKey, int keyLen, regDistPair *wideKNN, int wideK, int thinK, regDistPair *knn)
{
	int curDist = -1;
	// Build the initial heap
	for(int i = 0; i < thinK; ++i){
		knn[i].dist = numberOfDiff(queryKey,wideKNN[i].reg, keyLen);
		knn[i].regIndex = wideKNN[i].regIndex;
		knn[i].reg = wideKNN[i].reg;
		siftUp(0,i,knn);

	}
	// We have a max heap
	for(int i = thinK; i < wideK; ++i){
		curDist = numberOfDiff(queryKey,wideKNN[i].reg, keyLen);
		
		if(curDist < knn[0].dist){
			knn[0].dist = curDist;
			knn[0].regIndex = wideKNN[i].regIndex;
			knn[0].reg = wideKNN[i].reg;
			siftDown(0, thinK - 1, knn);
		}
	}
	// Heap Sort!
	for(int i = thinK - 1; i > 0; --i){
		swapPairs(0,i,knn);
		siftDown(0, i - 1, knn);
	}
}

void  moveRegs(
	regDistPair *knn, 
	int numRegs, 
	int keyLen, 
	kint * regs_out)
{
	for(int i = 0; i < numRegs; ++i){		
		memcpy(regs_out + i*keyLen, knn[i].reg,keyLen*sizeof(kint));
	}
}

void  moveIndexes(
	regDistPair *knn, 
	int numRegs, 
	int * indexes_out)
{
	for(int i = 0; i < numRegs; ++i){		
		indexes_out[i] = knn[i].regIndex;
	}
}