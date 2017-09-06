# Classifying Iris

![cover](./images/cover.png)

This is a companion sample project of the Iris tutorial that you can find from the main GitHub documentation site of Project "Vienna". Using the timeless [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), it walks you through the basics of Project "Vienna". 

- [Documentation site](https://github.com/Azure/ViennaDocs/blob/master/Documentation/Tutorial.md) for Microsoft internal dogfooders.
- [Documentation site](https://github.com/AzureMachineLearning/Project-Vienna-Private-Preview/blob/master/Documentation/Tutorial.md) for external private preview customers.

Enjoy!

## Quick CLI references
If you want to try quickly from the command line window launched from the _File_ menu:

Kick-off many local runs to observe the metrics emitted by each run in a graph.
```
$ python run.py
```

Run _iris_sklearn.py_ in local environment.
```
$ az ml experiment submit -c local iris_sklearn.py
```

Run _iris_sklearn.py_ in a local Docker container.
```
$ az ml experiment submit -c docker-python iris_sklearn.py
```

Run _iris_pyspark.py_ in a local Docker container.
```
$ az ml experiment submit -c docker-spark iris_pyspark.py
```

Create _myvm.compute_ file to point to a remote VM
```
$ az ml computetarget attach --name <myvm> --address <ip address or FQDN> --username <username> --password <pwd>
```

Run _iris_pyspark.py_ in a Docker container (with Spark) in a remote VM:
```
$ az ml experiment submit -c myvm iris_pyspark.py
```

Create _myhdi.compute_ to point to an HDI cluster
```
$ az ml computetarget attach --name <myhdi> --address <ip address or FQDN of the head node> --username <username> --password <pwd> --cluster
```

Run it in a remote HDInsight cluster:
```
$ az ml experiment submit -c myhdi iris_pyspark.py
```
