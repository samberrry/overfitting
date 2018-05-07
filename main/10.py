import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import subprocess

from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Prepares test data

NUMBER_OF_CLASS_A = 1800 # + class = classA
NUMBER_OF_CLASS_B = 1200 # o class = classB

cov = [[3, 0], [0, 3]]

mean1 = [10, 4]

mean2 = [5, 15]

mean3 = [15, 15]

# For 1200 points: NUMBER_OF_CLASS_B/3 = 400
x1, y1 = np.random.multivariate_normal(mean1, cov, NUMBER_OF_CLASS_B/3).T
x2, y2 = np.random.multivariate_normal(mean2, cov, NUMBER_OF_CLASS_B/3).T
x3, y3 = np.random.multivariate_normal(mean3, cov, NUMBER_OF_CLASS_B/3).T

# For 1200 points: (NUMBER_OF_CLASS_B/3)*30/100 = 120
trainingX1 = x1[:(NUMBER_OF_CLASS_B/3)*30/100]
trainingY1 = y1[:(NUMBER_OF_CLASS_B/3)*30/100]

trainingX2 = x2[:(NUMBER_OF_CLASS_B/3)*30/100]
trainingY2 = y2[:(NUMBER_OF_CLASS_B/3)*30/100]

trainingX3 = x3[:(NUMBER_OF_CLASS_B/3)*30/100]
trainingY3 = y3[:(NUMBER_OF_CLASS_B/3)*30/100]

testX1 = x1[(NUMBER_OF_CLASS_B/3)*30/100:]
testY1 = y1[(NUMBER_OF_CLASS_B/3)*30/100:]

testX2 = x2[(NUMBER_OF_CLASS_B/3)*30/100:]
testY2 = y2[(NUMBER_OF_CLASS_B/3)*30/100:]

testX3 = x3[(NUMBER_OF_CLASS_B/3)*30/100:]
testY3 = y3[(NUMBER_OF_CLASS_B/3)*30/100:]

# 540 points for + class = classA
# 360 points for o class = classB

classBxTraining = np.concatenate((trainingX1,trainingX2,trainingX3),axis= 0)
classByTraining = np.concatenate((trainingY1,trainingY2,trainingY3),axis= 0)

classBxTest = np.concatenate((testX1,testX2,testX3),axis= 0)
classByTest = np.concatenate((testY1,testY2,testY3),axis= 0)

samplx = np.random.uniform(low=0, high=20, size=(NUMBER_OF_CLASS_A,))
samply = np.random.uniform(low=0, high=20, size=(NUMBER_OF_CLASS_A,))

classAxTraining = samplx[:NUMBER_OF_CLASS_A * 30 / 100]
classAyTraining = samply[:NUMBER_OF_CLASS_A * 30 / 100]

classAxTest = samplx[NUMBER_OF_CLASS_A * 30 / 100:]
classAyTest = samply[NUMBER_OF_CLASS_A * 30 / 100:]

# =================DRAW Training Data=====================
# TEST DATA:
plt.plot(classAxTest, classAyTest, '+')
plt.plot(classBxTest, classByTest , 'o')
plt.title("Test Data")

plt.axis('equal')

plt.show()

# TRAINING DATA:
plt.plot(classAxTraining, classAyTraining, '+')
plt.plot(classBxTraining, classByTraining , 'o')
plt.title("Training Data")

plt.axis('equal')

plt.show()
# =================DRAW Training Data END==================

# =================Constructs Padnas Dataframe: Training
trainingDataA = {
    'x' :classAxTraining,
    'y': classAyTraining,
    'class' : '+'
}

dfA = pd.DataFrame(trainingDataA)

trainingDataB = {
    'x' :classBxTraining,
    'y': classByTraining,
    'class' : 'o'
}

dfB = pd.DataFrame(trainingDataB)

trainingData = dfA.append(dfB, ignore_index= True)

# shuffles data
trainingData = trainingData.sample(frac=1).reset_index(drop=True)
# =================Constructs Padnas Dataframe END

# =================Constructs Padnas Dataframe: Testing
testDataA = {
    'x' :classAxTest,
    'y': classAyTest,
    'class' : '+'
}

testDFA = pd.DataFrame(testDataA)

testDataB = {
    'x' :classBxTest,
    'y': classByTest,
    'class' : 'o'
}

testDFB = pd.DataFrame(testDataB)

testData = testDFA.append(testDFB, ignore_index= True)

# shuffles data
testData = testData.sample(frac=1).reset_index(drop=True)
# =================Constructs Padnas Dataframe END

#  encode the Classes to integers
def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

df2, targets = encode_target(trainingData, "class")

features = list(df2.columns[1:3])
y = df2["class"]
X = df2[features]

dt = DecisionTreeClassifier(min_samples_split=20)
dt.fit(X, y)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

visualize_tree(dt, features)
