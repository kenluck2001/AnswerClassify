#!/usr/bin/python
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

def logFile ( message, logfile="result.txt"):
    with open(logfile, 'a') as the_file:
        the_file.write(message + '\n')


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


#clf = MultinomialNB()

def selectFeatures (clf, X, Y):
    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(Y, 5),
                  scoring='accuracy')
    rfecv.fit(X, Y)
    lst = rfecv.get_support()
    indices = find(lst, True)
    return X[:, indices]


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
                '''
                elem = line[:-1]
                arrayRow = elem.split()
                if len (arrayRow) > 2:
                    idVal = arrayRow[0]
                    rowList = []
                    rowList.append (idVal)
                    for feat in arrayRow[1:]:
                        curFeatList = feat.split(':')
                        if len(curFeatList ) == 2:
                            rowList.append (curFeatList[1])
                    testingdf.loc[testingCnt] = rowList
                    testingCnt = testingCnt + 1
                '''
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

    return {'trainingdf': trainingdf, 'testingdf': testingdf}


def getData (fname):
    inputVal = dataFrameFromFile (fname)
    return inputVal['trainingdf'], inputVal['testingdf']


def getDataFromFrame (dframe, num):
    xCols = list( range(1, 24))
    dframe = dframe.head(n=num)
    X = dframe[xCols].values
    Y = dframe[[0]].values
    #fix the label in the right format
    Y = Y.ravel()
    return X, Y


if __name__ == '__main__':
    #fname = "test.txt"
    fname = "input00.txt"
    trainingdf, testingdf = getData (fname)

    X, Y = getDataFromFrame (trainingdf, num=1500)


    #Perform feature selection
    clf = MultinomialNB()
    xlist = selectFeatures (clf, X, Y)
    crossValidation(clf, X, Y)
    crossValidation(clf, xlist, Y)

    #find best K for nearest neighbours
    selectBestKNNUsingCrossValidation  ( X, Y )
    selectBestKNNUsingCrossValidation  ( xlist, Y )

    #perform model evaluation
    
    #K nearest neighbours
    #clf = KNeighborsClassifier() #default 5 neighbours
    #clf.fit( xlist, Y )
    #print clf.predict( X[1].reshape(1, -1) )
    #crossValidation(clf, xlist, Y)

    #Logistic regression
    #clf = LogisticRegression(C=1)
    #clf.fit( xlist, Y )
    #crossValidation(clf, xlist, Y)



    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc,
                       {'n_estimators': [10, 50, 100, 200, 300, 400, 500, 1000, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000]}, verbose=1)
    clf.fit(X,Y)
    print ("RandomForestClassifier")
    print(clf.best_score_)
    print(clf.best_params_)

    logFile ("RandomForestClassifier " + str (clf.best_score_) +" ! "+ str(clf.best_params_)+ "\n")

    clf.fit(xlist,Y)
    print ("RandomForestClassifier")
    print(clf.best_score_)
    print(clf.best_params_)

    logFile ("RandomForestClassifier " + str (clf.best_score_) +" ! "+ str(clf.best_params_)+ "\n")


    etc = ExtraTreesClassifier()
    clf = GridSearchCV(etc,
                       {'n_estimators': [10, 50, 100, 200, 300, 400, 500]}, verbose=1)
    clf.fit(X,Y)
    print ("ExtraTreesClassifier")
    print(clf.best_score_)
    print(clf.best_params_)

    logFile ("ExtraTreesClassifier " + str (clf.best_score_) +" ! "+ str(clf.best_params_)+ "\n")

    clf.fit(xlist,Y)
    print ("ExtraTreesClassifier")
    print(clf.best_score_)
    print(clf.best_params_)

    logFile ("ExtraTreesClassifier " + str (clf.best_score_) +" ! "+ str(clf.best_params_)+ "\n")

