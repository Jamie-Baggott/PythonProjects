#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on a windy day

@author: Jamie Baggott
@id: R00149982
@Cohort: SD3
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rnd

from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("weatherAUS.csv", encoding='utf8')


def Task1():

    dfc = df[['MinTemp', 'WindGustSpeed', 'Rainfall', 'RainTomorrow']].copy()

    dfc.loc[df['RainTomorrow'] == 'Yes', 'RainTomorrow'] = 2
    dfc.loc[df['RainTomorrow'] == 'No', 'RainTomorrow'] = 1

    flt = dfc[['MinTemp', 'WindGustSpeed', 'Rainfall', 'RainTomorrow']].dropna()

    flt = flt.fillna(flt.mean())

    # Part A
    X = (flt[['MinTemp', 'WindGustSpeed', 'Rainfall']])
    Y = flt[['RainTomorrow']]

    tree_clf = tree.DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X, Y)
    print("A Accuracy Score: ", tree_clf.score(X, Y))
    tree_clf = tree.DecisionTreeClassifier()

    a_cv_results = cross_validate(tree_clf, X, Y, cv=3, scoring='accuracy', return_train_score=True)

    print('A Training  ', a_cv_results['train_score'].mean())
    print('A Testing ....  ', a_cv_results['test_score'].mean())
    print("\n")

    a_X_train, a_X_test, a_Y_train, a_Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    c = 1
    a_mylistTr = []
    a_mylistTs = []
    index = range(1, 36)
    for n in range(35):
        clf = tree.DecisionTreeClassifier(max_depth=c, random_state=42)
        c = c + 1
        clf.fit(a_X_train, a_Y_train)

        a_mylistTr.append(clf.score(a_X_train, a_Y_train))
        a_mylistTs.append(clf.score(a_X_test, a_Y_test))

    plt.figure(figsize=(15, 10))

    x_ticks = np.arange(0, 35, 1)
    plt.xticks(x_ticks, rotation=45)

    y_ticks = np.arange(0, 1, .01)
    plt.yticks(y_ticks)

    plt.title("ACCURACY OF PREDICTING RAIN TOMORROW PART A", fontsize=15)
    plt.xlabel("Max Depth", fontsize=10)
    plt.ylabel("Accuracy Percentage", fontsize=10)

    plt.plot(index, a_mylistTr)
    plt.plot(index, a_mylistTs)
    plt.legend(['A Train', ' A Test'])
    plt.show()

    # Part B
    X = (flt[['MinTemp', 'WindGustSpeed']])
    Y = flt[['RainTomorrow']]

    tree_clf = tree.DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X, Y)
    print("B Accuracy Score: ", tree_clf.score(X, Y))
    tree_clf = tree.DecisionTreeClassifier()

    b_cv_results = cross_validate(tree_clf, X, Y, cv=3, scoring='accuracy', return_train_score=True)

    print('B Training  ', b_cv_results['train_score'].mean())
    print('B Testing ....  ', b_cv_results['test_score'].mean())
    print("\n")

    b_X_train, b_X_test, b_Y_train, b_Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    c = 1
    b_mylistTr = []
    b_mylistTs = []
    index = range(1, 36)
    for n in range(35):
        clf = tree.DecisionTreeClassifier(max_depth=c, random_state=42)
        c = c + 1
        clf.fit(b_X_train, b_Y_train)

        b_mylistTr.append(clf.score(b_X_train, b_Y_train))
        b_mylistTs.append(clf.score(b_X_test, b_Y_test))

    plt.figure(figsize=(15, 10))

    x_ticks = np.arange(0, 35, 1)
    plt.xticks(x_ticks, rotation=45)

    y_ticks = np.arange(0, 1, .01)
    plt.yticks(y_ticks)

    plt.title("ACCURACY OF PREDICTING RAIN TOMORROW PART B", fontsize=15)
    plt.xlabel("Max Depth", fontsize=10)
    plt.ylabel("Accuracy Percentage", fontsize=10)

    plt.plot(index, b_mylistTr)
    plt.plot(index, b_mylistTs)
    plt.legend(['B Train', 'B Test'])
    plt.show()

    # Part C
    X = (flt[['MinTemp', 'Rainfall']])
    Y = flt[['RainTomorrow']]

    tree_clf = tree.DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X, Y)
    print("C Accuracy Score: ", tree_clf.score(X, Y))
    tree_clf = tree.DecisionTreeClassifier()

    c_cv_results = cross_validate(tree_clf, X, Y, cv=3, scoring='accuracy', return_train_score=True)

    print('C Training  ', c_cv_results['train_score'].mean())
    print('C Testing ....  ', c_cv_results['test_score'].mean())
    print("\n")

    c_X_train, c_X_test, c_Y_train, c_Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    c = 1
    c_mylistTr = []
    c_mylistTs = []
    index = range(1, 36)
    for n in range(35):
        clf = tree.DecisionTreeClassifier(max_depth=c, random_state=42)
        c = c + 1
        clf.fit(c_X_train, c_Y_train)

        c_mylistTr.append(clf.score(c_X_train, c_Y_train))
        c_mylistTs.append(clf.score(c_X_test, c_Y_test))

    plt.figure(figsize=(15, 10))

    x_ticks = np.arange(0, 35, 1)
    plt.xticks(x_ticks, rotation=45)

    y_ticks = np.arange(0, 1, .01)
    plt.yticks(y_ticks)

    plt.title("ACCURACY OF PREDICTING RAIN TOMORROW PART C", fontsize=15)
    plt.xlabel("Max Depth", fontsize=10)
    plt.ylabel("Accuracy Percentage", fontsize=10)

    plt.plot(index, c_mylistTr)
    plt.plot(index, c_mylistTs)
    plt.legend(['C Train', 'C Test'])
    plt.show()

    # Part D
    X = (flt[['WindGustSpeed', 'Rainfall']])
    Y = flt[['RainTomorrow']]

    tree_clf = tree.DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X, Y)
    print("D Accuracy Score: ", tree_clf.score(X, Y))
    tree_clf = tree.DecisionTreeClassifier(max_depth=35)

    d_cv_results = cross_validate(tree_clf, X, Y, cv=3, scoring='accuracy', return_train_score=True)

    print('D Training  ', d_cv_results['train_score'].mean())
    print('D Testing ....  ', d_cv_results['test_score'].mean())
    print("\n")

    d_X_train, d_X_test, d_Y_train, d_Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    c = 1
    d_mylistTr = []
    d_mylistTs = []
    index = range(1, 36)
    for n in range(35):
        clf = tree.DecisionTreeClassifier(max_depth=c, random_state=42)
        c = c + 1
        clf.fit(d_X_train, d_Y_train)

        d_mylistTr.append(clf.score(d_X_train, d_Y_train))
        d_mylistTs.append(clf.score(d_X_test, d_Y_test))

    plt.figure(figsize=(15, 10))

    x_ticks = np.arange(0, 35, 1)
    plt.xticks(x_ticks, rotation=45)

    y_ticks = np.arange(0, 1, .01)
    plt.yticks(y_ticks)

    plt.title("ACCURACY OF PREDICTING RAIN TOMORROW PART D", fontsize=15)
    plt.xlabel("Max Depth", fontsize=10)
    plt.ylabel("Accuracy Percentage", fontsize=10)

    plt.plot(index, d_mylistTr)
    plt.plot(index, d_mylistTs)
    plt.legend(['D Train', ' D Test'])
    plt.show()

    plt.figure(figsize=(15, 10))

    x_ticks = np.arange(0, 35, 1)
    plt.xticks(x_ticks, rotation=45)

    y_ticks = np.arange(0, 1, .01)
    plt.yticks(y_ticks)

    plt.title("ACCURACY OF PREDICTING RAIN TOMORROW", fontsize=15)
    plt.xlabel("Max Depth", fontsize=10)
    plt.ylabel("Accuracy Percentage", fontsize=10)

    # plt.plot(index, a_mylistTr, label='A Training Score')
    plt.plot(index, a_mylistTs, label='A Test Score')
    # plt.plot(index, b_mylistTr, label='B Training Score')
    plt.plot(index, b_mylistTs, label='B Test Score')
    # plt.plot(index, c_mylistTr, label='C Training Score')
    plt.plot(index, c_mylistTs, label='C Test Score')
    # plt.plot(index, d_mylistTr, label='D Training Score')
    plt.plot(index, d_mylistTs, label='D Test Score')
    plt.legend()
    plt.show()


Task1()


# (a) Which dataset has a better accuracy in general?
# I would say that in general Dataset (d) WindGustSpeed, Rainfall and RainTomorrow had a better accuracy.

# (b) Which attribute has a more important role in predicting the RainTomorrow.
# Seeing as in general (b) was the  least accurate in general and the only one without Rainfall,
# as well (c) and (d) using only Rainfall and a second
# attribute being more accurate then I would say Rainfall is the most important role in prediciting it

# (c) What value approximately is an appropriate value for max_depth and why?
# It seems a value in or around 12-15 onwards would be most appropriate as the predcition is steady enough. You
# could see that aside from (a) there is no real drastic change from that point on.


def Task2():

    df["Pressure"] = df[['Pressure9am', 'Pressure3pm']].mean(axis=1)

    df["Humidity"] = df[['Humidity9am', 'Humidity3pm']].mean(axis=1)

    dfc = df[["Pressure", "Humidity", "RainToday"]].copy()

    dfc.loc[df['RainToday'] == 'Yes', 'RainToday'] = 2
    dfc.loc[df['RainToday'] == 'No', 'RainToday'] = 1

    dfc['Humidity'] = dfc['Humidity'].apply(pd.to_numeric, errors='coerce')
    dfc['Humidity'] = dfc['Humidity'].fillna(dfc['Humidity'].mean())

    dfc['Pressure'] = dfc['Pressure'].apply(pd.to_numeric, errors='coerce')
    dfc['Pressure'] = dfc['Pressure'].fillna(dfc['Pressure'].mean())

    flt = dfc[["Pressure", "Humidity", "RainToday"]].dropna()

    flt = flt.fillna(flt.mean())

    flt = flt.dropna()

    X = (flt[['Pressure', 'Humidity']])

    Y = flt[['RainToday']]

    models = [('KNN', KNeighborsClassifier()), ('DTC', tree.DecisionTreeClassifier()), ('NB', GaussianNB()),
              ('RFS', RandomForestClassifier())]

    # models.append(('SVM', SVC(kernel='linear')))
    names = []
    results = {}
    for name, model in models:
        cv_results = cross_validate(model, X, Y, cv=3, scoring='accuracy', return_train_score=True)

        results[name] = cv_results

    for models in results:
        print(models)
        print('Training  ', results[models]['train_score'].mean())
        print('Testing  ', results[models]['test_score'].mean())
        plt.figure(figsize=(15, 10))

        x_ticks = np.arange(0, 2, .1)
        plt.xticks(x_ticks)

        y_ticks = np.arange(0.5, 1, .01)
        plt.yticks(y_ticks)

        plt.title("ACCURACY OF DIFFERENT ALGORITHMS", fontsize=15)
        plt.xlabel(models.upper(), fontsize=10)
        plt.ylabel("Accuracy Percentage", fontsize=10)

        plt.plot(results[models]['train_score'], label='Training Score')
        plt.plot(results[models]['test_score'], label='Testing Score')
        plt.legend()
        plt.show()


Task2()


def Task3():

    dfc = df[['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'MinTemp']].copy()

    flt = dfc[['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'MinTemp']].dropna()

    data = flt.values[:, :-1]
    trans = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    data = trans.fit_transform(data)

    flt = flt.fillna(flt.mean())

    flt = flt.dropna()

    X = (flt[['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']])

    Y = flt[['MinTemp']]
    Y = Y.astype('int')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, Y_train)

    knn_cv = KNeighborsClassifier(n_neighbors=5)

    cv_scores = cross_val_score(knn_cv, X, Y, cv=2)

    print("X Prediction: ", knn.predict(X_test))
    print("Test Results: ", knn.score(X_test, Y_test))
    print("Train Results: ", knn.score(X_train, Y_train))

    print("Cross Validation: ", cv_scores)
    print("Average Cross Validation: {}".format(np.mean(cv_scores)))


Task3()


# What is the best number of bins for discretization over the following number of bins: 2, 3, 4, 5, 6 It would depend
# on the size and use that you're looking for. In general when working with a large dataset you'd want a large number
# of bins so it would make sense to use 5 or more when dealing with them in general


def Task4():

    dfc = df[['Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm']].copy()

    flt = dfc[['Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm']].dropna()

    # USE FOR 2 BINS FOR PART (A) ALONG WITH COMMENTED SECTION IN PLOT TOO
    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')

    # USE FOR 3 BINS FOR PART (D) ALONG WITH COMMENTED SECTION IN PLOT TOO
    # discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

    discretizer.fit(flt)
    dsFLT = discretizer.transform(flt)

    scalingObj = preprocessing.MinMaxScaler()
    newFLT = scalingObj.fit_transform(dsFLT)

    res = []
    for i in range(7):
        kmeans = KMeans(n_clusters=i + 1).fit(newFLT)
        res.append(kmeans.inertia_)
        clusterNum = i + 2
        print("Cluster Number %i :" % clusterNum, kmeans.inertia_)
    indexes = np.arange(2, 9)

    plt.figure(figsize=(15, 10))

    x_ticks = np.arange(2, 9, 1)
    plt.xticks(x_ticks)

    # USE FOR 2 BINS FOR PART (A) ALONG WITH COMMENTED SECTION IN DISCRETIZER TOO
    y_ticks = np.arange(0, 150000, 10000)

    # USE FOR 3 BINS FOR PART (D) ALONG WITH COMMENTED SECTION IN DISCRETIZER TOO
    # y_ticks = np.arange(0, 60000, 5000)

    plt.yticks(y_ticks)

    plt.title("TESTING PERFORMANCE OF NUMBER OF CLUSTERS", fontsize=15)
    plt.xlabel("Cluster Number", fontsize=10)
    plt.ylabel("Performance", fontsize=10)

    plt.plot(indexes, res, label='Cluster Performance')
    plt.legend()
    plt.show()


Task4()

# Use comment and explain how many clusters is more appropriate when the number of bins is two and when the number of
# bins is three

# I would say that when the number of bins is 2 then it would be optimal to use between 3 and 5 clusters and when bins
# are 3 that you should use between 2 and 4 clusters
