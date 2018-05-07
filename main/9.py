import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import subprocess

from sklearn.tree import DecisionTreeClassifier, export_graphviz

# This script makes works with sklearn

NUMBER_OF_O_CLASS= 400

cov = [[3, 0], [0, 3]]

mean1 = [10, 4]

mean2 = [5, 15]

mean3 = [15, 15]

x1, y1 = np.random.multivariate_normal(mean1, cov, NUMBER_OF_O_CLASS).T
x2, y2 = np.random.multivariate_normal(mean2, cov, NUMBER_OF_O_CLASS).T
x3, y3 = np.random.multivariate_normal(mean3, cov, NUMBER_OF_O_CLASS).T

x1 = x1[:120]
y1 = y1[:120]

x2 = x2[:120]
y2 = y2[:120]

x3 = x3[:120]
y3 = y3[:120]

samplx = np.random.uniform(low=0, high=20, size=(1800,))
samply = np.random.uniform(low=0, high=20, size=(1800,))

# 540 points for + class = classA
# 360 points for o class = classB

classBxTraining = np.concatenate((x1,x2,x3),axis= 0)
classByTraining = np.concatenate((y1,y2,y3),axis= 0)

classAxTraining = samplx[:540]
classAyTraining = samply[:540]

# =================DRAW Training Data=====================

plt.plot(classAxTraining, classAyTraining, '+')
plt.plot(classBxTraining, classByTraining , 'o')

plt.axis('equal')

# plt.show()

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
