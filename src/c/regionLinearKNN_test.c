#include <stdio.h>
#include <stdlib.h>
#include "regionLinearKNN.h" 

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

void testSimpleKNN(int K)
{

	kint regions[20];

	fillRegions(regions,20);
	printf("20 regions:\n");
	for(int i = 0; i < 20; i++){
		printf("[%d]:\t", i);
		printKey(regions + i,32);
	}
	printf("%d nearest neigbors:\n",K);
	regDistPair knn[19];
	knn[0].regIndex = -1;
	knn[0].dist = 0;
	knn[0].reg = regions;
	regKNN(regions, 1, regions + 1, 20 - 1, K - 1, knn + 1);
	for (int i = 0; i < K; ++i)
	{	
		printf("[%d] Dist: %f, index: %d \t",i,knn[i].dist,knn[i].regIndex);
		printKey(knn[i].reg,32);
	}
}

int main() 
{	
	for (int i = 1; i < 19; ++i)
	{
		testSimpleKNN(i);
	}
	return 0;
}