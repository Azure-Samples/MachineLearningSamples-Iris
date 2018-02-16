# Please make sure scikit-learn is included the conda_dependencies.yml file.
import pickle
import sys
import os

import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run

from plot_graphs import plot_iris

# initialize the logger
run_logger = get_azureml_logger() 

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

print('Python version: {}'.format(sys.version))
print()

# load Iris dataset from a DataPrep package as a pandas DataFrame
iris = run('iris.dprep', dataflow_idx=0, spark=False)
print ('Iris dataset shape: {}'.format(iris.shape))

# load features and labels
X, Y = iris[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values, iris['Species'].values

# add n more random features to make the problem harder to solve
# number of new random features to add
n = 40
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, n)]

# split data 65%-35% into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])

print("Regularization rate is {}".format(reg))

# log the regularization rate
run_logger.log("Regularization Rate", reg)

# train a logistic regression model on the training set
clf1 = LogisticRegression(C=1/reg).fit(X_train, Y_train)
print (clf1)

# evaluate the test set
accuracy = clf1.score(X_test, Y_test)
print ("Accuracy is {}".format(accuracy))

# log accuracy which is a single numerical value
run_logger.log("Accuracy", accuracy)

# calculate and log precision, recall, and thresholds, which are list of numerical values
y_scores = clf1.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(Y_test, y_scores[:,1],pos_label='Iris-versicolor')
run_logger.log("Precision", precision)
run_logger.log("Recall", recall)
run_logger.log("Thresholds", thresholds)

print("")
print("==========================================")
print("Serialize and deserialize using the outputs folder.")
print("")

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()

# load the model back from the 'outputs' folder into memory
print("Import the model from model.pkl")
f2 = open('./outputs/model.pkl', 'rb')
clf2 = pickle.load(f2)

# predict on a new sample
X_new = [[3.0, 3.6, 1.3, 0.25]]
print ('New sample: {}'.format(X_new))

# add random features to match the training data
X_new_with_random_features = np.c_[X_new, random_state.randn(1, n)]

# score on the new sample
pred = clf2.predict(X_new_with_random_features)
print('Predicted class is {}'.format(pred))

try:
    import matplotlib
    # plot confusion matrix and ROC curve
    plot_iris(clf1, X, Y)
    print("Confusion matrix and ROC curve plotted. See them in Run History details page.")
except ImportError:
    print("matplotlib not found so no plots produced.")
    print("Please install it from command-line window by typing \"pip install matplotlib\".")
    
