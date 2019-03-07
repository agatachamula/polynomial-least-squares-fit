# Polynomial least squares fit

This repository consists of implementation of polynomial least square fit with testing module and visualization.
There are two models - with and without regularization

## Prerequisites

This project uses numpy for computation, pickle for data managment and matplotlib for visualization of results.

## Content.py

File consisting of needed methods:

* Mean square error
* Design matrix builder
* Least squares
* Regularized least squares - with regularization parameter lambda
* Model selection - based on mean square error
* Regularized model selection - based on mean square error

Inputs and outputs of all methods are described in the comments in the code.

## Test.py

File consisting of unittests for methods in content.py. Expected result for all tests is "ok".

## Main.py

From main the tests are run and least square fit is preformed. 
Methods are run for number of traning samples 

```
N=[8, 50]
```
and regularization parameters lambda

```
lambdas = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300]
```
and polynomial degree
```
M=[0,1,2,3,4,5,6,7]
```
Results are presented on graphs prepared with matplotlib.

##Acknowledgments

This project was prepared as a part of Machine Learning course in Wrocław University of Technology.


