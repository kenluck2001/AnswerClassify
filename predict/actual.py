#!/usr/bin/python
from __future__ import division
import pandas as pd
import numpy as np


from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV

featureSelectModel = [] #models requiring feature selection
wholeFeatureModel = []  #models not requiring feature selection


def crossValidation(clf, X, Y, num=None):
    '''
        num: can be number of trees or nearest neighbours
    '''
    scores = []
    cv = StratifiedKFold(Y, n_folds=5)
    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]
        clf.fit( X_train, y_train )
        scores.append(clf.score( X_test, y_test ))
    if num:
        print("Classifier: " + str (clf.__str__ )+ "\t Mean(scores)= " + str (np.mean(scores) ) + "\tStddev(scores)= " + str (np.std(scores))+ "\t Number of neighbours / trees= " + str (num) + "\n")
        logFile ("Classifier: " + str (clf.__str__ )+ "\t Mean(scores)= " + str (np.mean(scores) ) + "\tStddev(scores)= " + str (np.std(scores))+ "\t Number of neighbours / trees= " + str (num) + "\n")
    else:
        print("Classifier: " + str (clf.__str__ )+ "\t Mean(scores)= " + str (np.mean(scores) ) + "\tStddev(scores)= " + str (np.std(scores)) + "\n")
        logFile ("Classifier: " + str (clf.__str__ )+ "\t Mean(scores)= " + str (np.mean(scores) ) + "\tStddev(scores)= " + str (np.std(scores)) + "\n")



def selectBestKNNUsingCrossValidation ( X, Y ):
    '''
        This can work for nearest neighbours
    '''
    for k in range (2, 100):
        clf = KNeighborsClassifier(n_neighbors=k)
        crossValidation(clf, X, Y, num=k )


def find(lst, elem):
    return [i for i, x in enumerate(lst) if x == elem ]



def selectFeatures (clf, X, Y):
    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(Y, 5),
                  scoring='accuracy')
    rfecv.fit(X, Y)
    lst = rfecv.get_support()
    indices = find(lst, True)
    return X[:, indices], indices


def dataFrameFromFile (fname):
    testing = False

    #Training
    trainingCols = tuple(['id'] + range(24))
    trainingdf = pd.DataFrame(columns=trainingCols)
    trainingCnt = 0

    #Testing
    testingCols = tuple(['id'] + range(23))
    testingdf = pd.DataFrame(columns=testingCols)
    testingCnt = 0

    with open(fname) as f:
        next(f)
        for line in f:
            if  len (line.split() ) == 1:
                testing = True
            if testing:
                elem = line[:-1]        
                arrayRow = elem.split()
                if len (arrayRow) > 2:
                    idVal = arrayRow[0]
                    rowList = []
                    if idVal:
                        rowList.append (idVal)
                        for feat in arrayRow[1:]:
                            curFeatList = feat.split(':')
                            if len(curFeatList ) == 2:
                                rowList.append (curFeatList[1])
                    if len (rowList) == 24:
                        testingdf.loc[testingCnt] = rowList
                        testingCnt = testingCnt + 1
            else:
                elem = line[:-1]
                arrayRow = elem.split()
                if len (arrayRow) > 2:
                    idVal = arrayRow[0]
                    rowList = []
                    rowList.append (idVal)
                    for feat in arrayRow[1:]:
                        curFeatList = feat.split(':')
                        if len(curFeatList ) == 1:
                            rowList.append (curFeatList[0])
                        if len(curFeatList ) == 2:
                            rowList.append (curFeatList[1])
                    trainingdf.loc[trainingCnt] = rowList
                    trainingCnt = trainingCnt + 1

    return  trainingdf, testingdf



def getDataFromFrameForTraining (dframe, num):
    xCols = list( range(1, 24))
    dframe = dframe.head(n=num)
    X = dframe[xCols].values
    Y = dframe[[0]].values
    #fix the label in the right format
    Y = Y.ravel()
    return X, Y

def getDataFromFrameForTesting (dframe, num):
    xCols = list( range(0, 23))
    dframe = dframe.head(n=num)
    X = dframe[xCols].values
    idlist = dframe[['id']].values
    return X, idlist



def makEnsemble( X, xlist, Y ):
    #naive bayes
    clf = MultinomialNB()
    clf.fit( xlist, Y )
    featureSelectModel.append (clf)

    #K nearest neighbours
    clf = KNeighborsClassifier()
    clf.fit( xlist, Y )
    featureSelectModel.append (clf)

    #Logistic regression
    clf = LogisticRegression(C=1)
    clf.fit( xlist, Y )
    featureSelectModel.append (clf)

    #random forest
    clf  = RandomForestClassifier(n_estimators = 400)
    clf.fit( X, Y )
    wholeFeatureModel.append (clf)

    #extra forest
    clf = ExtraTreesClassifier(n_estimators = 400)
    clf.fit( X, Y )
    wholeFeatureModel.append (clf)

    #decision forest
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
    clf.fit( X, Y )
    wholeFeatureModel.append (clf)

    #gradient boosting
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
                  'learning_rate': 0.01}
    clf = GradientBoostingClassifier(**params)
    clf.fit( X, Y )
    wholeFeatureModel.append (clf)



def enspredict (Xval, indices):
    '''
        blend models using majority voting scheme
    '''
    totalLabelist = []
    for ind in range (len(Xval)):
        labelist = []
        for model in featureSelectModel:
            label = model.predict( Xval[:, indices ][ind].reshape(1, -1) )
            labelist.append (np.asscalar (label) )

        for model in wholeFeatureModel:
            label = model.predict( Xval[ind].reshape(1, -1) )
            labelist.append (np.asscalar (label) )


        votedLabel = max ( set (labelist), key=labelist.count  )
        totalLabelist.append (votedLabel)

    return totalLabelist


def accuracy(actual, pred):
    total = len (actual)
    count = 0
    for acc, pval in zip (actual, pred):
        if acc == pval:
            count = count + 1
    return count / total

 

def logResult ( idlist, pred, logfile="new-output.txt"):
    with open(logfile, 'a') as the_file:
        for idval, ylabel in zip( idlist, pred ):
            the_file.write(str (idval) +"  "  + str(ylabel) + '\n')


if __name__ == '__main__':
    #fname = "test.txt"
    fname = "input00.txt"
    trainingdf, testingdf = dataFrameFromFile (fname)
    #Obtain training data

    trainingdfCnt = trainingdf.shape[0] #gives number of row count
    testingdfCnt = testingdf.shape[0] 

    Xtrain, Ytrain = getDataFromFrameForTraining (trainingdf, num=trainingdfCnt )

    #Perform feature selection
    clf = MultinomialNB()
    xlist, indices = selectFeatures (clf, Xtrain, Ytrain)

    #train models
    makEnsemble( Xtrain, xlist, Ytrain )

    Xtest, idlist = getDataFromFrameForTesting (testingdf, num=testingdfCnt)

    #make predictions
    preds = enspredict (Xtest, indices)

    #write result to file
    logResult ( sum(idlist.tolist(), []) , preds)


