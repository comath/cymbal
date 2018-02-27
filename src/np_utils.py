import numpy as np

def removeClasses(trainingData,trainingLabels,classes):
	currentLabels = np.argmax(trainingLabels,axis=1)
	trainingIndexes = np.arange(trainingData.shape[0], dtype=np.int32)
	for c in classes:
		trainingIndexes = np.extract(currentLabels != c, trainingIndexes)
		currentLabels = currentLabels[trainingIndexes]
						
	trainingLabels = np.delete(trainingLabels,classes,1)
	return trainingData[trainingIndexes], trainingLabels[trainingIndexes]

def leaveClasses(trainingData,trainingLabels,classes):
	currentIndexes = np.array([],dtype=np.int32)
	currentLabels = np.argmax(trainingLabels,axis=1)	
	trainingIndexes = np.arange(trainingData.shape[0], dtype=np.int32)

	for c in classes:
		trainingIndexes = np.extract(currentLabels == c, trainingIndexes)
		currentIndexes = np.concatenate([currentIndexes,trainingIndexes])
	
	allClasses = list(range(trainingLabels.shape[1]))
	removeClasses = list(set(allClasses) - set(classes))
	trainingLabels = np.delete(trainingLabels,removeClasses,1)
	return trainingData[currentIndexes], trainingLabels[currentIndexes]