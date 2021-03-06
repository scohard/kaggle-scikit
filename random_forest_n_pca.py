#!/usr/bin/env python

import csv_io

from numpy import ravel
import numpy as np

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import zero_one_loss
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split

num_pca = 12

def drawLearningCurve(model, x_train, y_train, x_test, y_test, num_points = 50):
# adapted from http://sachithdhanushka.blogspot.de/2013/09/learning-curve-generator-for-learning.html
    
    train_error = np.zeros(num_points)
    crossval_error = np.zeros(num_points)
    
    sizes = np.linspace(2, len(x_train), num=num_points).astype(int)
    for i,size in enumerate(sizes):
         
        #getting the predicted results of the model
        model.fit(x_train[:size], y_train[:size])
         
        #compute the validation error
        y_pred = model.predict(x_test[:size])
        crossval_error[i] = zero_one_loss(y_test[:size], y_pred, normalize=True)
         
        #compute the training error
        y_pred = model.predict(x_train[:size])
        train_error[i] = zero_one_loss(y_train[:size], y_pred, normalize=True)

    #draw the plot
    print crossval_error
    print train_error
    fig,ax = plt.subplots()
    ax.plot(sizes,crossval_error,lw = 2, label='cross validation error')
    ax.plot(sizes,train_error, lw = 4, label='training error')
    ax.set_xlabel('cross val error')
    ax.set_ylabel('rms error')
    ax.legend(loc = 0)
    ax.set_title('Learning Curve' )
    return fig

def main(strat = False, visualization = False):
    #read in the training file
    X = csv_io.read_data("data/train.csv")
    target = ravel(csv_io.read_data("data/trainLabels.csv"))
    realtest = csv_io.read_data("data/test.csv")
    print len(realtest)

    #pca
    pca = PCA(n_components=num_pca)
    pca.fit(X)
    train = pca.transform(X)
    test_transformed = pca.transform(realtest)
    print('performed pca')

    # random forest code
    rf = RandomForestClassifier(n_estimators=20, min_samples_split=2, n_jobs=-1, random_state=42, oob_score=True)
    if strat:
        print "stratified cross-validation on shuffled data"    
        # adapted from http://stackoverflow.com/a/8281241
        crossval = []
        for i in range(strat):
            X, y = shuffle(train, target, random_state=i)
            skf = StratifiedKFold(y, 10)
            crossval.append([min(cross_val_score(rf, X, y, cv=skf)), np.median(cross_val_score(rf, X, y, cv=skf)), max(cross_val_score(rf, X, y, cv=skf))]) 
        print crossval

    if visualization:
        print "preparing visualization"
        data_train, data_test, target_train, target_test = train_test_split(train, target, test_size=0.20, random_state=42)
        plot1 = drawLearningCurve(rf, data_train, target_train, data_test, target_test)
        pp = PdfPages('figures/learningCurve.pdf')
        pp.savefig(plot1)
        pp.close()

    print('fitting the model')
    rf.fit(train, target)
    # run model against test data
    predicted_class = rf.predict(test_transformed)
    print predicted_class[0:9]
    print(len(predicted_class))

    print('Writing output')
    predicted_class = ["%d,%d" % (i+1, predicted_class[i]) for i in range(len(predicted_class))]
    print predicted_class[0:9]
    print(len(predicted_class))
    csv_io.write_delimited_file("results/random_forest_solution.csv", predicted_class, header=['Id', 'Solution'])

    print ('Finished. Exiting.')

if __name__ == "__main__":
    main(visualization=True, strat=1)
