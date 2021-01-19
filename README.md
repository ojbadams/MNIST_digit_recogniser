# MNIST_digit_recogniser


In the current state MNIST Digit File contains an analysis of the data in the MNIST csv file (downloaded from Kaggle) with use of principle component analysis (PCA) to reduce the dataset down before a use of K-nearest neighbour to succesfully identify images. 

The Accuracy currently sits at a fairly respectible 97%, there is an analysis of the some of the missclassifications (tpr, fpr) and it seems that the current algorithm is having a difficult time identifying "9". 

The second file, boosted_df, is code to try and boost the size of the dataset using data augmentation - this is a work in progress as processing time is too large. 

TODO: 
- Write up work so far explaining results 
- Succesfully expand the MNIST dataset using the second file
- Implement and compare results to other classifiers - (probably a neural network and decision tree)
