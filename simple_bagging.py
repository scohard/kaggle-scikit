#!/usr/bin/env python

import csv_io
import numpy as np



        
def main():
    np.random.seed(42)
    #read in the training file
    modelone = np.asarray(csv_io.read_data("results/gmm_pca12_6.csv",header=True))
    modeltwo = np.asarray(csv_io.read_data("results/random_forest_solution-12pca-4.csv",header=True))
    modelthree= np.asarray(csv_io.read_data("results/svm_pca12_5.csv",header=True))
    bagmodel = np.column_stack((modelone[:,1], modeltwo[:,1], modelthree[:,1]))
    bagsum = bagmodel.sum(axis=1)
    predicted_class = np.zeros(bagsum.shape)
    predicted_class[bagsum >=2] = 1
    
    predicted_class = ["%d,%d" % (i+1, predicted_class[i]) for i in range(len(predicted_class))]
    print predicted_class[0:9]
    print(len(predicted_class))
    csv_io.write_delimited_file("results/bagging_solution_7.csv", predicted_class, header=['Id', 'Solution'])

if __name__ == "__main__":
    main()
