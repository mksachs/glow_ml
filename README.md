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




python -m glow_ml_train.train predict_accident --holdout 0.8 --algorithm lr

pytest -v

