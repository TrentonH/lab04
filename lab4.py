__author__ = 'Trenton'
#include
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import random
import numpy as np


#part 1
print("i read the full requirements for part one")

#part 2

class TreeModle:
    def __init__(self,X_train, y_train, n_neighbors):
        self.X_train = X_train
        self.y_train = y_train


class decisionTreeClassifier:
    def __init__(self, n_neighbors):
        pass
    def fit(self, X_train, y_train):
        return TreeModle(X_train, y_train)

#part 3
# reading in voting-records

def checkEqual(lst):
   return lst[1:] == lst[:-1]

#calculate entropy given p == the prcentage
#part 5
def calc_entropy(p):
    if p!=0:
        return -p *np.log2(p)
    else:
        return 0



def buildTree(x_train, y_train, myTree):
    subx = []
    suby = []
    #need some good logic here

    #base case
    if(checkEqual(suby)):
        return
    else:
        buildTree(subx, suby, myTree)

def printTree(myTree):
    for x in myTree:
        print(x)
    for y in myTree[x]:
        print(y,':',myTree[x][y])


#tht

def main ():
    #pulls in the data and gives it columns
    VO = pd.io.parsers.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",header=None,na_values = "?" )

    obj_def = VO.select_dtypes(include=['object']).copy()

    VO.columns = ["party", "handicapped-infants", "water-project-cost-sharings", "adoption-of-the-budget-resolution",
                  "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
                  "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
                  "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports",
                  "export-administration-act-south-africa"]
    # drop row of the  the na values for the purpos of testing
    #VO = VO.dropna()

    obj_def = obj_def.dropna()
    VONP = obj_def.as_matrix()

    VOTargets =[]
    for x in VONP:
        VOTargets.append(x[16])
    VONP = np.delete(VONP, 16, 1)

    X_train, X_test, y_train, y_test = train_test_split(VONP, VOTargets, test_size=0.3)

    #part 4
    myTree = {}




if __name__ == "__main__":
    main()