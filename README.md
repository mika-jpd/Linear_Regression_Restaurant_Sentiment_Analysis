# Linear Regression Restaurant Sentiment Analysis

## Table of Contens
* [Description](#description)
* [Dataset](#dataset)
* [Setup](#setup)
* [Run code](#run-code)

## Description
Linear Regression Machine Learning model to predict whether a review is positive or negative. It correctly predicts the correct label with an accuracy of 86%.

## Technologies
Project is created with:
* Python version: 3.9.1
* NumPy library version : 1.20.0
* Pandas library version : 1.2.2

## Dataset
The dataset has been made such that each feature is a categorical feature (0, 1) representing the presence or absence of words used in restaurant reviews. Common words such as "the", "a" etc... are not categorized.

Each row represents a single point (restaurant review) and each column an individual feature. Since all features are binary features, each point's columns will hold either a 1 (presence of the word represented by the ith feature) or 0 (absence of the word represented by the ith feature).

## Setup
Download the .py files, training_dataset, validation_dataset, and weights file. Place them in a single file or project file.

## Run Code

Add the following to the class file:

```
  x = logistic_regression("train_dataset.tsv", "validation_dataset.tsv",
                        k, alpha, tolerance)
  x.fit()
  x.predict()
```
where,

* k = max iterations
* alpha = the learning rate
* tolerance = the tolerance rate
      
Suggested parameters are:
* k = 5000
* alpha = 0.01
* tolerance = 0.0005

Enjoy!
