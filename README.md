
![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 7 | Module 3 - Supervised Machine Learning 

## Authors
### Vito Coquet & Ildem Sanli 

## Introduction

The goal of this project is to practise supervised learning using Netflix data. We need to create the model for the rating prediction. 
Each group will need to research and implement the defined supervised machine learning methods.

## About the Dataset

A dataset of 8451 rows and 14 columns containing characteristics of movies on Netflix from IMDB, such as genre, director, rating, and so on.
Some columns have numeric types but most of the columns have object types because the cells in them contain a list of values. There is a total of 8508 NaN values.

## Data Cleaning 

We started by dropping the columns that we thought were not relevant for our models.
Then we converted the ‘rating’, ‘vote’ and ‘runtime’ columns types to numeric.
Dropped a few outliers, made the ‘kind’ column more uniform.
We decided to keep only the first country of each cell in the corresponding column, same for the ‘genre’ column.
We kept only the 10 most occurring values in the ‘country’ column.
And finally we scaled the runtime column.

## Models 

We used the following models :

-RidgeClassifier : Calssifier based on ridge regression. It converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case).

-SVC : C-support vector classification is a class of Support Vector Machines(SVM). SVMs divide the datasets into number of classes by generating a hyperplane.

-CategoricalNB : Categorical Naïve Bayes is a probabilistic classifier based on Bayes Theorem with a strong independence assumption between the features.
It is suitable for classification with discrete features that are categorically distributed.

-ExtraTreesClassifier : Ensemble method composed of a large number of decision trees, where the final decision is obtained taking into account the prediction of every tree. 

We obtained the best results with Extra Trees Classifier. 

## Repo Structure

Our repo is organised as follows:

Data: Original and cleaned datasets in CSV format

Code: Data cleaning, Exploratory Data Analysis and Model Implementation

Presentation: Slides explaining the project and showing the results
