# Please make sure scikit-learn is included the conda_dependencies.yml file.

import pickle
import sys
import os

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from azureml.logging import get_azureml_logger

# initialize the logger
run_logger = get_azureml_logger() 

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

print ('Python version: {}'.format(sys.version))
print ()

# load Iris dataset
iris = load_iris()
print ('Iris dataset shape: {}'.format(iris.data.shape))

# load features and labels
X, Y = iris.data, iris.target

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])

print("Regularization rate is {}".format(reg))

# log the regularization rate
run_logger.log("Regularization Rate", reg)

# train a logistic regression model
clf1 = LogisticRegression(C=1/reg).fit(X, Y)
print (clf1)

accuracy = clf1.score(X, Y)
print ("Accuracy is {}".format(accuracy))

# log accuracy
run_logger.log("Accuracy", accuracy)

print("")
print("==========================================")
print("Serialize and deserialize using the native share folder: {0}".format(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']))
print("")

# serialize the model on disk in the private share folder. 
# note this folder is NOT tracked by run history, but it survives across runs on the same compute context.
print ("Export the model to model.pkl in the native shared folder")
f = open(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + 'model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()

# load the model back from the private share folder into memory
print("Import the model from model.pkl in the native shared folder")
f2 = open(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + 'model.pkl', 'rb')
clf2 = pickle.load(f2)

# predict a new sample
X_new = [[3.0, 3.6, 1.3, 0.25]]
print ('New sample: {}'.format(X_new))
pred = clf2.predict(X_new)
print('Predicted class is {}'.format(pred))
