#!/usr/bin/env python

import csv_io

from numpy import ravel
import numpy as np
from sklearn.decomposition import PCA

from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM
#inspired by http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_classifier.html

num_pca = 12


        
def main(pca=False):
    np.random.seed(42)
    #read in the training file
    data = np.asarray(csv_io.read_data("data/train.csv"))
    labels = ravel(csv_io.read_data("data/trainLabels.csv"))
    realtest = csv_io.read_data("data/test.csv")
    print len(realtest)
    n_classes = len(np.unique(labels))
    
    if pca:
        #pca
        pca = PCA(n_components=num_pca)
        pca.fit(data)
        data = pca.transform(data)
        realtest = pca.transform(realtest)
        print('performed pca')
        
        
    skf = StratifiedKFold(labels, n_folds=10)
    classifiers = dict((covar_type, GMM(n_components=n_classes,
    covariance_type=covar_type, init_params='wc', n_iter=20))
    for covar_type in ['spherical', 'diag', 'tied', 'full'])
    
    # Try GMMs using different types of covariances.

    for index, (name, classifier) in enumerate(classifiers.iteritems()):
        print "========"
        print name, classifier
    
        for train_index, test_index in skf:
            X_train = data[train_index]
            y_train = labels[train_index]
            X_test = data[test_index]
            y_test = labels[test_index]
            # random forest code
            
            

                       
            # Since we have class labels for the training data, we can
            # initialize the GMM parameters in a supervised manner.
            classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
            for i in xrange(n_classes)])

            # Train the other parameters using the EM algorithm.
            classifier.fit(X_train)



            y_train_pred = classifier.predict(X_train)
            train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
            print 'Train accuracy: %.1f' % train_accuracy

            y_test_pred = classifier.predict(X_test)
            test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
            print 'Test accuracy: %.1f' % test_accuracy

        print ('Now using full GMM for prediction')
        
        classifier = GMM(n_components=n_classes,
            covariance_type='full', init_params='wc', n_iter=20)
        classifier.means_ = np.array([data[labels == i].mean(axis=0)
        for i in xrange(n_classes)])
        # Train the other parameters using the EM algorithm.
        classifier.fit(data)

        predicted_class = classifier.predict(realtest)
        predicted_class = ["%d,%d" % (i+1, predicted_class[i]) for i in range(len(predicted_class))]
        print predicted_class[0:9]
        print(len(predicted_class))
        csv_io.write_delimited_file("results/gmm_pca12_corrected_8.csv", predicted_class, header=['Id', 'Solution'])
        
        
        print ('Finished. Exiting.')

if __name__ == "__main__":
    main(pca=True)
