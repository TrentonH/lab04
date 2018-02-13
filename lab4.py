__author__ = 'Trenton'
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import copy


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


class node:
    columns = []
    child1 = ""
    child0 = ""
    is_leaf = True
    is_true = 1
    x_values = []
    y_values = []
    posible_values = []
    def __init__(self,posible_values, name_of_node, x_values, y_values):
        self.column = name_of_node
        self.posible_values = posible_values
        self.x_values = x_values
        self.y_values = y_values
        if len(posible_values) == 0:
            self.is_leaf = True
            i = 0
            j = 0
            for x in self.y_values:
                if x == 0:
                    i += 1
                else:
                    j += 1
            if i > j:
                self.is_true = 0
        else:
            self.is_leaf = False

    def set_column(self, columnName):
        self.column = columnName
    def set_child0(self, node):
        self.child0 = node
    def set_child1(self, node):
        self.child1 = node
    def set_leaf(self, posible_values):
        if len(posible_values) == 0:
            self.is_leaf = True
        else:
            self.is_leaf = False

#part 3
# reading in voting-records

def checkEqual(lst):
   return lst[1:] == lst[:-1]

#calculate entropy given p == the prcentage
#part 5

#here is another one
def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mutual_information(y, x):#book function
    res = entropy(y)
    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])
    return res

def is_pure(s):
    return len(set(s)) == 1




def buildTree(x_train, y_train, columns):
    #Create a root node for the tree
    thisNode = node(columns,"temp", x_train, y_train)
    #If all examples are 1, Return the single-node tree Root, with label = 1.
    #If all examples are 0, Return the single-node tree Root, with label = 0.
    if checkEqual(thisNode.y_values):
        if len(thisNode.y_values) != 0:
            if(thisNode.y_values[0]) == 0:
                thisNode.is_true = 0

        else:
            thisNode.is_leaf = True
            thisNode.is_true = 1
        return thisNode

    #If number of predicting columns is empty, then Return the single node tree Root,
    #with label = most common value of the target attribute in the examples.

        #this is done when i inishalize thisNode
        return ThisNode
    #Otherwise Begin
        #A ? The column that best classifies examples.
        idor = 0
        bestEntropy = 1000.00000
        column_to_remove = None
        while idor < 15:
            subToCheck = []
            for x , y in zip(x_train, y_train):
                if x[idor] == 1:
                    subToCheck.append(y)
                    tempE =  entropy(subToCheck)
                if bestEntropy > tempE:
                    bestEntropy = TempE
                    column_to_remove = idor
            idor += 1
        #creat new train x train y and columns pass into its slef
            sub_setx1 = []
            sub_sety1 = []
            sub_setx0 = []
            sub_sety0 = []
            for c, d in zip(x_train , y_train):
                if c[column_to_remove] == 1:
                    sub_setx1.append(c)
                    sub_sety1.append(d)
                else:
                    sub_setx0.append(c)
                    sub_sety0.append(d)
            sub_setc = []
            for a in columns:
                if a != column_to_remove:
                    sub_setc.append(a)

            #populate  new trains with data
            thisNode.child1 = buildTree(sub_setx1,sub_sety1,sub_setc)
            thisNode.child0 = buildTree(sub_setx0,sub_sety0,sub_setc)
        #do this for the 1 side and the zero side, set as children
        #if one side does not have any values set child to None
    #End
    #Return Root
    return thisNode




#pass in "root" first time
def printTree(treeValue, myTree):
    subTree = None
    #base case this is a leaf
    if myTree.is_leaf:
        print(treeValue)
        print(myTree.column)
        return
    #base case there is a left and right node
    subTree = copy.deepcopy(myTree.child1)
    printTree("1" , subTree)
    subTree = copy.deepcopy(myTree.child0)
    printTree("0" , subTree)
    print(treeValue)
    print(myTree.column)
    return

def predictID3(X_test, columns, tree, answers, child1Or0, isLeaf):
    thisAnswer = None
    if tree.is_leaf == True:
        isLeaf = True
    if isLeaf and (tree.is_leaf or child1Or0 == 1 or child1Or0 == 0):
        return child1Or0
    #loop tell leaf node if its a 1 tree anser is 1 else anser is 0
    #fined assosiated x value
    else:
        idor = 0
        location = 0
        #fined the number of the first column
        for x in columns:
            if tree.column == x:
                location = idor
            idor += 1
        if X_test[location] == 1:
            predictID3(X_test, columns, tree, answers, 1, isLeaf)
        else:
            predictID3(X_test, columns, tree, answers, 0, isLeaf)







def main ():
    #pulls in the data and gives it columns
    VO = pd.io.parsers.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",header=None,na_values = "?" )

    #1  obj_def = VO.select_dtypes(include=['object']).copy()

    VO.columns = ["party", "handicapped-infants", "water-project-cost-sharings", "adoption-of-the-budget-resolution",
                  "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
                  "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
                  "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports",
                  "export-administration-act-south-africa"]
    VO = VO.dropna()
    VO["party"] = VO["party"].replace (["republican"], 1)
    VO["party"] = VO["party"].replace (["democrat"], 0)
    VO["handicapped-infants"] = VO["handicapped-infants"].replace (["y"], 1)
    VO["handicapped-infants"] = VO["handicapped-infants"].replace (["n"], 0)
    VO["water-project-cost-sharings"] = VO["water-project-cost-sharings"].replace (["y"], 1)
    VO["water-project-cost-sharings"] = VO["water-project-cost-sharings"].replace (["n"], 0)
    VO["adoption-of-the-budget-resolution"] = VO["adoption-of-the-budget-resolution"].replace (["y"], 1)
    VO["adoption-of-the-budget-resolution"] = VO["adoption-of-the-budget-resolution"].replace (["n"], 0)
    VO["physician-fee-freeze"] = VO["physician-fee-freeze"].replace (["y"], 1)
    VO["physician-fee-freeze"] = VO["physician-fee-freeze"].replace (["n"], 0)
    VO["el-salvador-aid"] = VO["el-salvador-aid"].replace (["y"], 1)
    VO["el-salvador-aid"] = VO["el-salvador-aid"].replace (["n"], 0)
    VO["religious-groups-in-schools"] = VO["religious-groups-in-schools"].replace (["y"], 1)
    VO["religious-groups-in-schools"] = VO["religious-groups-in-schools"].replace (["n"], 0)
    VO["anti-satellite-test-ban"] = VO["anti-satellite-test-ban"].replace (["y"], 1)
    VO["anti-satellite-test-ban"] = VO["anti-satellite-test-ban"].replace (["n"], 0)
    VO["aid-to-nicaraguan-contras"] = VO["aid-to-nicaraguan-contras"].replace (["y"], 1)
    VO["aid-to-nicaraguan-contras"] = VO["aid-to-nicaraguan-contras"].replace (["n"], 0)
    VO["mx-missile"] = VO["mx-missile"].replace (["y"], 1)
    VO["mx-missile"] = VO["mx-missile"].replace (["n"], 0)
    VO["immigration"] = VO["immigration"].replace (["y"], 1)
    VO["immigration"] = VO["immigration"].replace (["n"], 0)
    VO["synfuels-corporation-cutback"] = VO["synfuels-corporation-cutback"].replace (["y"], 1)
    VO["synfuels-corporation-cutback"] = VO["synfuels-corporation-cutback"].replace (["n"], 0)
    VO["education-spending"] = VO["education-spending"].replace (["y"], 1)
    VO["education-spending"] = VO["education-spending"].replace (["n"], 0)
    VO["superfund-right-to-sue"] = VO["superfund-right-to-sue"].replace (["y"], 1)
    VO["superfund-right-to-sue"] = VO["superfund-right-to-sue"].replace (["n"], 0)
    VO["crime"] = VO["crime"].replace (["y"], 1)
    VO["crime"] = VO["crime"].replace (["n"], 0)
    VO["duty-free-exports"] = VO["duty-free-exports"].replace (["y"], 1)
    VO["duty-free-exports"] = VO["duty-free-exports"].replace (["n"], 0)
    VO["export-administration-act-south-africa"] = VO["export-administration-act-south-africa"].replace (["y"], 1)
    VO["export-administration-act-south-africa"] = VO["export-administration-act-south-africa"].replace (["n"], 0)

    VONP = VO.as_matrix()
    #1  obj_def = obj_def.dropna()
    #1  VONP = obj_def.as_matrix()

    VOTargets =[]
    for x in VONP:
        VOTargets.append(x[16])
    VONP = np.delete(VONP, 16, 1)


    x = "{'n': 9, 'y': 7} {'n': 8, 'y': 8} {'n': 8, 'y': 8} {'n': 9, 'y': 7} {'n': 9, 'y': 7} {'n': 9, 'y': 7} {'n': 8, 'y': 8} {'n': 6, 'y': 10} {'n': 8, 'y': 8} {'n': 8, 'y': 8} {'n': 7, 'y': 9}  {'n': 8, 'y': 8} {'n': 9, 'y': 7} {'n': 8, 'y': 8} {'n': 8, 'y': 8}{'n': 7, 'y': 9} {'n': 7, 'y': 9}"
    #default = classes['y','y','y','y','y','y','y','n','y','y','n','y','y''y','y','n','n']
    X_train, X_test, y_train, y_test = train_test_split(VONP, VOTargets, test_size=0.3)
    rootTree  = copy.deepcopy(buildTree(X_train,y_train,VO.columns))
    printTree("root", rootTree)
    #prints the % predicted correct
    answers = []
    count = 0
    answers = predictID3(X_test, VO.columns, rootTree, answers, "", False)
    for s , x in zip (answers, y_test):
        if s == x:
            count += 1
    print (count / len(y_test))


if __name__ == "__main__":
    main()