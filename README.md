# Glow ML

Glow ML is a lightweight machine learning training and deployment framework. It is composed of two packages: 
**glow_ml** which is a flask microservice API to deploy models and deliver predictions, and **glow_ml_train**
which is a command line tool to train models and prepare them for deployment. Additionally, **Glow ML** 
includes a utility to output stats for deployed models: **model_test**.

## Installation

Glow ML requires python 3.6 or greater. The installation instructions here are for MacOSX using [homebrew](https://brew.sh "Homebrew"), [pyenv](https://github.com/pyenv/pyenv "pyenv") and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv "pyenv-virtualenv").

### One: Install Homebrew

```/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"```

[Detailed installation instructions](https://brew.sh "Homebrew")

### Two: Install pyenv and pyenv-virtualenv

```brew install pyenv```

[Detailed installation instructions](https://github.com/pyenv/pyenv#homebrew-on-mac-os-x)

```brew install pyenv-virtualenv```

Add the following to your shell profile:

```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

[Detailed installation instructions](https://github.com/pyenv/pyenv-virtualenv)


### Three: Install python 3.6 or greater

To see a list of all available versions do this:

```pyenv install --list```

Glow ML was developed using the vanillia version of python `3.7.0` (don't use versions that have 
`anaconda-` or `pypy-` in them). 

To install python:

```pyenv install 3.7.0```

You can now check to see what versions are installed using:

```pyenv versions```

### Four: Create a virtual python environment for Glow ML

```pyenv virtualenv 3.7.0 glow_ml```

Activate this virtual environment:

```pyenv activate glow_ml```

Your shell prompt should now look someting like this:

```(glow_ml) The-Queen-of-Cups:```

To deactivate the virtual environment do this:

```pyenv deactivate```

### Five: Install python dependencies

From the active virtual environment, install using `pip`:

```pip install --upgrade -r requirements.txt```

Congratulations! You are ready to go.

## Running tests

### Unit tests

**Glow ML** uses `pytest` for unit test discovery and execution.  To run tests do this:

```pytest -v```

### Model tests

In addition to unit tests, **Glow ML** also includes a model testing package. This assumes that there is a
serialized model present in the `glow_ml/models` directory, and that there is testing data in the `glow_ml_train/data`
directory. A model and testing data is included in the github repo so you can run the model testing package. The model
testing package runs in two modes: the first loads the model directly into the testing package, the second sends all
testing data through the API. The first mode can be run like this (from the main `glow_ml` project directory):

```python -m model_tests predict_accident```

The second mode requires that an API server is running. See [Running the server](#running-the-server) for instructions
on how to do this. Once the server is running, the model tests can be run through the API by simply passing the the 
API URL using the `--test_url` command line option.

```python -m model_tests predict_accident --test_url http://127.0.0.1:5000/api/v1.0/predict_accident```

## Running the server






export FLASK_ENV=development
export FLASK_APP=glow_ml:glow_ml_app
flask run



python -m glow_ml_train.train predict_accident --holdout 0.8 --algorithm lr

pytest -v

