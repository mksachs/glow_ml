# Glow ML

Glow ML is a lightweight machine learning training and deployment framework. It is composed of two packages: 
**glow_ml** which is a flask microservice API to deploy models and deliver predictions, and **glow_ml_train**
which is a command line tool to train models and prepare them for deployment. Additionally, **Glow ML** 
includes a utility to output stats for deployed models: **model_test**.

## Installation

python -m glow_ml_train.train predict_accident --holdout 0.8 --algorithm lr

pytest -v

