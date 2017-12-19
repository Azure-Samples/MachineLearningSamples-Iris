# Classifying Iris

This is a companion sample project of the Azure Machine Learning [QuickStart](https://docs.microsoft.com/azure/machine-learning/preview/quickstart-installation) and [Tutorials](https://docs.microsoft.com/azure/machine-learning/preview/tutorial-classifying-iris-part-1). Using the timeless [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), it walks you through the basics of preparing dataset, creating a model and deploying it as a web service.

![cover](./docs/iris.png)

## QuickStart
Select `local` as the execution environment, and `iris_sklearn.py` as the script, and click **Run** button.  You can also set the _Regularization Rate_ by entering `0.01` in the **Arguments** control.  Changing the _Regularization Rate_ has an impact on the accuracy of the model, giving interesting results to explore.

## Exploring results
After running, you can check out the results in **Run History**.  Exploring the **Run History** will allow you to see the correlation between the parameters you entered and the accuracy of the models.  You can get individual run details by clicking a run in the **Run History** report or clicking the name of the run on the Jobs Panel to the right.  In this sample you will have richer results if you have `matplotlib` installed.

## Quick CLI references
If you want to try exercising the Iris sample from the command line, here are some things to try:

First, launch the Command Prompt or Powershell from the **File** menu. Then enter the following commands:

```
# first let's install matplotlib locally
$ pip install matplotlib

# log in to Azure if you haven't done so
$ az login

# kick off many local runs sequentially
$ python run.py
```

Run `iris_sklearn.py` Python script in local Python environment.
```
$ az ml experiment submit -c local iris_sklearn.py
```

Run `iris_sklearn.py` Python script in a local Docker container.
```
$ az ml experiment submit -c docker-python iris_sklearn.py
```

Run `iris_spark.py` PySpark script in a local Docker container.
```
$ az ml experiment submit -c docker-spark iris_spark.py
```

Create `myvm` run configuration to point to a Docker container on a remote VM
```
$ az ml computetarget attach remotedocker --name myvm --address <ip address or FQDN> --username <username> --password <pwd>

# prepare the environment
$ az ml experiment prepare -c myvm
```

Run `iris_spark.py` PySpark script in a Docker container (with Spark) in a remote VM:
```
$ az ml experiment submit -c myvm iris_spark.py
```

Create `myhdi` run configuration to point to an HDI cluster
```
$ az ml computetarget attach cluster --name myhdi --address <ip address or FQDN of the head node> --username <username> --password <pwd> 

# prepare the environment
$ az ml experiment prepare -c myhdi
```

Run in a remote HDInsight cluster:
```
$ az ml experiment submit -c myhdi iris_spark.py
```
