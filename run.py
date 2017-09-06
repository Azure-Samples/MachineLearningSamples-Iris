# run iris_sklearn.py with descending regularization rates
# run this with just "python run.py". It will fail if you run using az ml execute.

import os

# set regularization rate as an argument 
reg = 10
while reg > 0.005:
    os.system('az ml experiment submit -c local ./iris_sklearn.py {}'.format(reg))
    # cut regularization rate to half
    reg = reg / 2
    
