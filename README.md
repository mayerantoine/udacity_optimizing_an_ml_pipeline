# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

The best performing model was a VotingEnsemble which achieve a performance of 91.6 % accuracy  compare to the  LogistiRegression model with the best paramaters who
only achieve a 90.6 % performance accurancy.

## Scikit-learn Pipeline

We downloaded the data from a web link, clean it and one hot encoded some key features , split in train and test with 25% of the data in testing,
And then we train the data using Hyperdrive with RandomSampling. We only focus our hyperparament fine tuning on two key parameters of Logistic Regression :
    * C  : Inverse of regularization strength; must be a positive float.
    *  max_iter : Maximum number of iterations taken for the solvers to converge.

We used Random sampling search because it should more faster, allow more coverage of the search space and parameter values are chosen from a set of discrete values or a distribution over a continuous range.

We used a Bandit Policy early termination, because  it automatically terminate poorly performing runs with an early termination policy and terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.This Early termination improves computational efficiency.

## AutoML
We downloaded the data from a web link, clean it and one hot encoded some key features , split in train and test with 25% of the data in testing.
Then we created a dataset that we passed to Azure Auto ML.


<img src="creating-and-optimizing-an-ml-pipeline.png"
     alt="creating-and-optimizing-an-ml-pipeline"
     style="float: left; margin-right: 10px;" />

## Pipeline comparison
Auto ML find the  best performing model  which was a VotingEnsemble which achieve a performance of 91.6 % accuracy  compare to the  LogistiRegression model with the best paramaters who only achieve a 90.6 % performance accurancy.

This was predictable because the Auto ML search a larger set of ensemble models as the first approach only hyper parameters of the baseline Logistic Regression model.

## Future work

Next steps would be to investigate the Voting Ensemble and Hyperparameters to see how we could improve performace. Also the classes were imbalances, we could also 
look at the impact of over sampling or under sampling the data on performance in conjunction with other feature engineering techniques.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
