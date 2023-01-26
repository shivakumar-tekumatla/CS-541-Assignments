import numpy as np 
def trainModel():
    return  
def testModel():
    return 
def doCrossValidation (D, k, h):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    for fold in range(k):
        # Get all indexes for this fold
        testIdxs = idxs[fold,:]
        # Get all the other indexes
        trainIdxs = idxs[list(set(allIdxs) - set(testIdxs)),:].flatten()
        # Train the model on the training data
        model = trainModel(D[trainIdxs], h)
        # Test the model on the testing data
        accuracies.append(testModel(model, D[testIdxs]))
    return np.mean(accuracies)


def doDoubleCrossValidation (D, k, H):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    acc =0 
    h_star = np.random.choice(H) # Initially taking a random hyper parameter set as the best one 
    for fold in range(k):
        # Get all indexes for this fold
        testIdxs = idxs[fold,:]
        # Get all the other indexes
        trainIdxs = idxs[list(set(allIdxs) - set(testIdxs)),:].flatten()
        for h in H:
            new_acc = doCrossValidation (D[trainIdxs], k, h)
            if  new_acc > acc:
                acc =  new_acc 
                h_star = h 
        # Train the model on the training data
        model = trainModel(D[trainIdxs], h_star)
        # Test the model on the testing data
        accuracies.append(testModel(model, D[testIdxs]))
    return np.mean(accuracies)
