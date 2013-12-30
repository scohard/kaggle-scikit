#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
import csv_io
from numpy import ravel


def main():
    #read in the training file
    train = csv_io.read_data("data/train.csv")
    target = ravel(csv_io.read_data("data/trainLabels.csv"))

    realtest = csv_io.read_data("data/test.csv")
    print len(realtest)

    # random forest code
    rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1, random_state=1, oob_score=True)
    # fit the training data
    print('fitting the model')
    rf.fit(train, target)

    # run model against test data
    predicted_probs = rf.predict_proba(realtest)
    predicted_class = rf.predict(realtest)
    print predicted_class[1:10]
    print(len(predicted_class))

    predicted_probs = ["%f" % x[1] for x in predicted_probs]
    predicted_class = ["%d,%d" % (i+1, predicted_class[i]) for i in range(len(predicted_class))]
    print predicted_class[0:9]
    print(len(predicted_class))

    csv_io.write_delimited_file("results/random_forest_solution.csv", predicted_class, header=['Id', 'Solution'])

if __name__ == "__main__":
    main()
