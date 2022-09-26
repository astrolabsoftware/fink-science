# Building a science module

## Before starting

In order to design a Fink Science module, you will need to have Apache Spark installed (2.4.x), and fink-broker cloned somewhere, with `$FINK_HOME` defined in your path.

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

You can copy and edit this tutorial to create your own science module. Then edit the configuration file provided in this folder for the shell in Fink to use jupyter:

```bash
# in fink.conf.shell
# Pyspark driver: None, Ipython, or Jupyter-notebook
# Note: for Jupyter on a cluster, you might need to specify the options
# --no-browser --port=<PORT>, and perform port redirection when ssh-ing.
PYSPARK_DRIVER_PYTHON=`which jupyter-notebook`
```

Finally launch it using the `fink_shell`:

```bash
$FINK_HOME/bin/fink_shell -c fink.conf.shell
```

You should have access to jupyter notebooks with Spark inside! In the current tutorial, we just aggregate magnitude measurements contained in alerts, and compute the difference between measurements. Probably meaningless (we do not split by filter bands), the purpose of the tutorial is to show you basic tools to build your module.

## Testing your module (broker)

We use `doctest`. See available science modules to have an idea of how we test our modules.

## What is next?

Once you are happy with your science module, open a PR and we will review it before merging it.
