# Building a science module

## Before starting

In order to design a Fink Science module, you will need to have Apache Spark installed (3.x), and fink-utils installed:

```bash\
# Install fink-utils
pip install fink-utils --upgrade

# Install Apache Spark
SPARK_VERSION=3.1.3
wget --quiet https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz
tar -xf spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz
rm spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz
```

and put these lines in your ~/.bash_profile:

```bash
export SPARK_HOME=/path/to/spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}
export PATH=$PATH:$SPARK_HOME/bin
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python
```

## Structure of a science module

Fink science modules are contained in `fink_science/`. They all have the same structure:

```
.
├── README.md
├── __init__.py
├── processor.py
└── utilities.py
```

- `__init__.py`: empty (module)
- `processor.py`: contains the Spark pandas UDF(s)
- `utilities.py`: anything else you need to make your UDF to work

## Testing your module (tutorial)

Just launch:

```bash
PYSPARK_DRIVER_PYTHON=`which jupyter-notebook` `which pyspark`
```

You should have access to jupyter notebooks with Spark inside! In the current tutorial, we just aggregate magnitude measurements contained in alerts, and compute the difference between measurements. Probably meaningless (we do not split by filter bands), the purpose of the tutorial is to show you basic tools to build your module.

## Testing your module (broker)

We use `doctest`. See available science modules to have an idea of how we test our modules.

## What is next?

Once you are happy with your science module, open a PR and we will review it before merging it.
