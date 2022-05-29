
# Evaluation of Diagnostic Models

Welcome to the second assignment of course 1. In this assignment, we will be working with the results of the X-ray classification model we developed in the previous assignment. In order to make the data processing a bit more manageable, we will be working with a subset of our training, and validation datasets. We will also use our manually labeled test dataset of 420 X-rays.  

As a reminder, our dataset contains X-rays from 14 different conditions diagnosable from an X-ray. We'll evaluate our performance on each of these classes using the classification metrics we learned in lecture.

**By the end of this assignment you will learn about:**

1. Accuracy
1. Prevalence
1. Specificity & Sensitivity
1. PPV and NPV
1. ROC curve and AUCROC (c-statistic)
1. Confidence Intervals

## Table of Contents

- [1. Packages](#1)
- [2. Overview](#2)
- [3. Metrics](#3)
    - [3.1 - True Positives, False Positives, True Negatives and False Negatives](#3-1)
        - [Exercise 1 - true positives, false positives, true negatives, and false negatives](#ex-1)
    - [3.2 - Accuracy](#3-2)
        - [Exercise 2 - get_accuracy](#ex-2)
    - [3.3 Prevalence](#3-3)
        - [Exercise 3 - get_prevalence](#ex-3)
    - [3.4 Sensitivity and Specificity](#3-4)
        - [Exercise 4 - get_sensitivity and get_specificity](#ex-4)
    - [3.5 PPV and NPV](#3-5)
        - [Exercise 5 - get_ppv and get_npv](#ex-5)
    - [3.6 ROC Curve](#3-6)
- [4. Confidence Intervals](#4)
- [5. Precision-Recall Curve](#5)
- [6. F1 Score](#6)
- [7. Calibration](#7)

<a name='1'></a>
## 1. Packages

In this assignment, we'll make use of the following packages:
- [numpy](https://docs.scipy.org/doc/numpy/) is a popular library for scientific computing
- [matplotlib](https://matplotlib.org/3.1.1/contents.html) is a plotting library compatible with numpy
- [pandas](https://pandas.pydata.org/docs/) is what we'll use to manipulate our data
- [sklearn](https://scikit-learn.org/stable/index.html) will be used to measure the performance of our model


Run the next cell to import all the necessary packages as well as custom util functions.


```python
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  

import util
from public_tests import *
from test_utils import *
```

<a name='2'></a>
## 2. Overview

We'll go through our evaluation metrics in the following order.

- Metrics
  - TP, TN, FP, FN
  - Accuracy
  - Prevalence
  - Sensitivity and Specificity
  - PPV and NPV
  - AUC
- Confidence Intervals

Let's take a quick peek at our dataset. The data is stored in two CSV files called `train_preds.csv` and `valid_preds.csv`. We have precomputed the model outputs for our test cases. We'll work with these predictions and the true class labels throughout the assignment.


```python
train_results = pd.read_csv("data/train_preds.csv")
valid_results = pd.read_csv("data/valid_preds.csv")

# the labels in our dataset
class_labels = ['Cardiomegaly',
 'Emphysema',
 'Effusion',
 'Hernia',
 'Infiltration',
 'Mass',
 'Nodule',
 'Atelectasis',
 'Pneumothorax',
 'Pleural_Thickening',
 'Pneumonia',
 'Fibrosis',
 'Edema',
 'Consolidation']

# the labels for prediction values in our dataset
pred_labels = [l + "_pred" for l in class_labels]
```

Extract the labels (y) and the predictions (pred).


```python
y = valid_results[class_labels].values
pred = valid_results[pred_labels].values
```

Run the next cell to view them side by side.


```python
# let's take a peek at our dataset
valid_results[np.concatenate([class_labels, pred_labels])].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cardiomegaly</th>
      <th>Emphysema</th>
      <th>Effusion</th>
      <th>Hernia</th>
      <th>Infiltration</th>
      <th>Mass</th>
      <th>Nodule</th>
      <th>Atelectasis</th>
      <th>Pneumothorax</th>
      <th>Pleural_Thickening</th>
      <th>...</th>
      <th>Infiltration_pred</th>
      <th>Mass_pred</th>
      <th>Nodule_pred</th>
      <th>Atelectasis_pred</th>
      <th>Pneumothorax_pred</th>
      <th>Pleural_Thickening_pred</th>
      <th>Pneumonia_pred</th>
      <th>Fibrosis_pred</th>
      <th>Edema_pred</th>
      <th>Consolidation_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.256020</td>
      <td>0.266928</td>
      <td>0.312440</td>
      <td>0.460342</td>
      <td>0.079453</td>
      <td>0.271495</td>
      <td>0.276861</td>
      <td>0.398799</td>
      <td>0.015867</td>
      <td>0.156320</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.382199</td>
      <td>0.176825</td>
      <td>0.465807</td>
      <td>0.489424</td>
      <td>0.084595</td>
      <td>0.377318</td>
      <td>0.363582</td>
      <td>0.638024</td>
      <td>0.025948</td>
      <td>0.144419</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.427727</td>
      <td>0.115513</td>
      <td>0.249030</td>
      <td>0.035105</td>
      <td>0.238761</td>
      <td>0.167095</td>
      <td>0.166389</td>
      <td>0.262463</td>
      <td>0.007758</td>
      <td>0.125790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.158596</td>
      <td>0.259460</td>
      <td>0.334870</td>
      <td>0.266489</td>
      <td>0.073371</td>
      <td>0.229834</td>
      <td>0.191281</td>
      <td>0.344348</td>
      <td>0.008559</td>
      <td>0.119153</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.536762</td>
      <td>0.198797</td>
      <td>0.273110</td>
      <td>0.186771</td>
      <td>0.242122</td>
      <td>0.309786</td>
      <td>0.411771</td>
      <td>0.244666</td>
      <td>0.126930</td>
      <td>0.342409</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



To further understand our dataset details, here's a histogram of the number of samples for each label in the validation dataset:


```python
plt.xticks(rotation=90)
plt.bar(x = class_labels, height= y.sum(axis=0));
```


![png](output_12_0.png)


It seem like our dataset has an imbalanced population of samples. Specifically, our dataset has a small number of patients diagnosed with a `Hernia`.

<a name='3'></a>
## 3. Metrics

<a name='3-1'></a>
### 3.1 True Positives, False Positives, True Negatives and False Negatives

The most basic statistics to compute from the model predictions are the true positives, true negatives, false positives, and false negatives. 

As the name suggests
- True Positive (TP): The model classifies the example as positive, and the actual label also positive.
- False Positive (FP): The model classifies the example as positive, **but** the actual label is negative.
- True Negative (TN): The model classifies the example as negative, and the actual label is also negative.
- False Negative (FN): The model classifies the example as negative, **but** the label is actually positive.

We will count the number of TP, FP, TN and FN in the given data.  All of our metrics can be built off of these four statistics. 

Recall that the model outputs real numbers between 0 and 1.
* To compute binary class predictions, we need to convert these to either 0 or 1. 
* We'll do this using a threshold value $th$.
* Any model outputs above $th$ are set to 1, and below $th$ are set to 0. 

All of our metrics (except for AUC at the end) will depend on the choice of this threshold. 

<a name='ex-1'></a>
### Exercise 1 -  true positives, false positives, true negatives and false negatives

Fill in the functions to compute the TP, FP, TN, and FN for a given threshold below. 

The first one has been done for you.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def true_positives(y, pred, th=0.5):
    """
    Count true positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TP (int): true positives
    """
    TP = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th

    # compute TP
    TP = np.sum((y == 1) & (thresholded_preds == 1))
    
    return TP

def true_negatives(y, pred, th=0.5):
    """
    Count true negatives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TN (int): true negatives
    """
    TN = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # compute TN
    TN = np.sum((y == 0) & (thresholded_preds == 0))
    
    ### END CODE HERE ###
    
    return TN

def false_positives(y, pred, th=0.5):
    """
    Count false positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FP (int): false positives
    """
    FP = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # compute FP
    FP = np.sum((y == 0) & (thresholded_preds == 1))
    
    ### END CODE HERE ###
    
    return FP

def false_negatives(y, pred, th=0.5):
    """
    Count false positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FN (int): false negatives
    """
    FN = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # compute FN
    FN = np.sum((y == 1) & (thresholded_preds == 0))
    
    ### END CODE HERE ###
    
    return FN
```


```python
### do not modify this cell    
get_tp_tn_fp_fn_test(true_positives, true_negatives, false_positives, false_negatives)    
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_test</th>
      <th>preds_test</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.8</td>
      <td>TP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.7</td>
      <td>TP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.4</td>
      <td>TN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.3</td>
      <td>TN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.2</td>
      <td>TN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.5</td>
      <td>FP</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0.6</td>
      <td>FP</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0.7</td>
      <td>FP</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0.8</td>
      <td>FP</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.1</td>
      <td>FN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.2</td>
      <td>FN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.3</td>
      <td>FN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0.4</td>
      <td>FN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0.0</td>
      <td>FN</td>
    </tr>
  </tbody>
</table>
</div>


    Your functions calcualted: 
        TP: 2
        TN: 3
        FP: 4
        FN: 5
        
    [92m All tests passed.
    [92m All tests passed.
    [92m All tests passed.
    [92m All tests passed.


##### Expected output

```Python
Your functions calcualted: 
    TP: 2
    TN: 3
    FP: 4
    FN: 5
```
```
 All tests passed.
 All tests passed.
 All tests passed.
 All tests passed.
```

Run the next cell to see a summary of evaluative metrics for the model predictions for each class. 


```python
util.get_performance_metrics(y, pred, class_labels)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Prevalence</th>
      <th>Sensitivity</th>
      <th>Specificity</th>
      <th>PPV</th>
      <th>NPV</th>
      <th>AUC</th>
      <th>F1</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>16</td>
      <td>814</td>
      <td>169</td>
      <td>1</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>20</td>
      <td>869</td>
      <td>103</td>
      <td>8</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>99</td>
      <td>690</td>
      <td>196</td>
      <td>15</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>1</td>
      <td>743</td>
      <td>255</td>
      <td>1</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>114</td>
      <td>543</td>
      <td>265</td>
      <td>78</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>40</td>
      <td>789</td>
      <td>158</td>
      <td>13</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>28</td>
      <td>731</td>
      <td>220</td>
      <td>21</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>64</td>
      <td>657</td>
      <td>249</td>
      <td>30</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>24</td>
      <td>785</td>
      <td>183</td>
      <td>8</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>24</td>
      <td>713</td>
      <td>259</td>
      <td>4</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>14</td>
      <td>661</td>
      <td>320</td>
      <td>5</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>10</td>
      <td>725</td>
      <td>261</td>
      <td>4</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>15</td>
      <td>767</td>
      <td>213</td>
      <td>5</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>36</td>
      <td>658</td>
      <td>297</td>
      <td>9</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



Right now it only has TP, TN, FP, FN. Throughout this assignment we'll fill in all the other metrics to learn more about our model performance.

<a name='3-2'></a>
### 3.2 - Accuracy

Let's use a threshold of .5 for the probability cutoff for our predictions for all classes and calculate our model's accuracy as we would normally do in a machine learning problem. 

$$accuracy = \frac{\text{true positives} + \text{true negatives}}{\text{true positives} + \text{true negatives} + \text{false positives} + \text{false negatives}}$$

<a name='ex-2'></a>
### Exercise 2 - get_accuracy

Use this formula to compute accuracy below:

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>Remember to set the value for the threshold when calling the functions.</li>
</ul>
</p>


```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_accuracy(y, pred, th=0.5):
    """
    Compute accuracy of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        accuracy (float): accuracy of predictions at threshold
    """
    accuracy = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # get TP, FP, TN, FN using our previously defined functions
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)

    # Compute accuracy using TP, FP, TN, FN
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    ### END CODE HERE ###
    
    return accuracy
```


```python
### do not modify this cell    
get_accuracy_test(get_accuracy)
```

    Test Case:
    
    Test Labels:	   [1 0 0 1 1]
    Test Predictions:  [0.8 0.8 0.4 0.6 0.3]
    Threshold:	   0.5
    Computed Accuracy: 0.6 
    
    [92m All tests passed.


#### Expected output:

```Python
Test Case:

Test Labels:	   [1 0 0 1 1]
Test Predictions:  [0.8 0.8 0.4 0.6 0.3]
Threshold:	     0.5
Computed Accuracy: 0.6 
```
```
 All tests passed.
```

Run the next cell to see the accuracy of the model output for each class, as well as the number of true positives, true negatives, false positives, and false negatives.


```python
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Prevalence</th>
      <th>Sensitivity</th>
      <th>Specificity</th>
      <th>PPV</th>
      <th>NPV</th>
      <th>AUC</th>
      <th>F1</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>16</td>
      <td>814</td>
      <td>169</td>
      <td>1</td>
      <td>0.83</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>20</td>
      <td>869</td>
      <td>103</td>
      <td>8</td>
      <td>0.889</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>99</td>
      <td>690</td>
      <td>196</td>
      <td>15</td>
      <td>0.789</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>1</td>
      <td>743</td>
      <td>255</td>
      <td>1</td>
      <td>0.744</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>114</td>
      <td>543</td>
      <td>265</td>
      <td>78</td>
      <td>0.657</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>40</td>
      <td>789</td>
      <td>158</td>
      <td>13</td>
      <td>0.829</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>28</td>
      <td>731</td>
      <td>220</td>
      <td>21</td>
      <td>0.759</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>64</td>
      <td>657</td>
      <td>249</td>
      <td>30</td>
      <td>0.721</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>24</td>
      <td>785</td>
      <td>183</td>
      <td>8</td>
      <td>0.809</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>24</td>
      <td>713</td>
      <td>259</td>
      <td>4</td>
      <td>0.737</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>14</td>
      <td>661</td>
      <td>320</td>
      <td>5</td>
      <td>0.675</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>10</td>
      <td>725</td>
      <td>261</td>
      <td>4</td>
      <td>0.735</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>15</td>
      <td>767</td>
      <td>213</td>
      <td>5</td>
      <td>0.782</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>36</td>
      <td>658</td>
      <td>297</td>
      <td>9</td>
      <td>0.694</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



If we were to judge our model's performance based on the accuracy metric, we would say that our model is not very accurate for detecting the `Infiltration` cases (accuracy of 0.657) but pretty accurate for detecting `Emphysema` (accuracy of 0.889). 

**But is that really the case?...**

Let's imagine a model that simply predicts that any patient does **Not** have `Emphysema`, regardless of patient's measurements. Let's calculate the accuracy for such a model.


```python
get_accuracy(valid_results["Emphysema"].values, np.zeros(len(valid_results)))
```




    0.972



As you can see above, such a model would be 97% accurate! Even better than our deep learning based model. 

But is this really a good model? Wouldn't this model be wrong 100% of the time if the patient actually had this condition?

In the following sections, we will address this concern with more advanced model measures - **sensitivity and specificity** - that evaluate how well the model predicts positives for patients with the condition and negatives for cases that actually do not have the condition.

<a name='3-3'></a>
### 3.3 - Prevalence

Another important concept is **prevalence**. 
* In a medical context, prevalence is the proportion of people in the population who have the disease (or condition, etc). 
* In machine learning terms, this is the proportion of positive examples. The expression for prevalence is:

$$prevalence = \frac{1}{N} \sum_{i} y_i$$

where $y_i = 1$ when the example is 'positive' (has the disease).

<a name='ex-3'></a>
### Exercise 3 - get_prevalence

Let's measure prevalence for each disease:

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>
    You can use <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html" > np.mean </a> to calculate the formula.</li>
    <li>Actually, the automatic grader is expecting numpy.mean, so please use it instead of using an equally valid but different way of calculating the prevalence. =) </li>
</ul>
</p>



```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_prevalence(y):
    """
    Compute prevalence.

    Args:
        y (np.array): ground truth, size (n_examples)
    Returns:
        prevalence (float): prevalence of positive cases
    """
    prevalence = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    prevalence = np.mean(y, axis=0)
    
    ### END CODE HERE ###
    
    return prevalence
```


```python
### do npt modify this cell    
get_prevalence_test(get_prevalence)
```

    Test Case:
    
    Test Labels:	      [1 0 0 1 1 0 0 0 0 1]
    Computed Prevalence:  0.4 
    
    [92m All tests passed.


#### Expected output:

```Python
Test Case:

Test Labels:	      [1 0 0 1 1 0 0 0 0 1]
Computed Prevalence:  0.4  
```
```
 All tests passed.
```


```python
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Prevalence</th>
      <th>Sensitivity</th>
      <th>Specificity</th>
      <th>PPV</th>
      <th>NPV</th>
      <th>AUC</th>
      <th>F1</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>16</td>
      <td>814</td>
      <td>169</td>
      <td>1</td>
      <td>0.83</td>
      <td>0.017</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>20</td>
      <td>869</td>
      <td>103</td>
      <td>8</td>
      <td>0.889</td>
      <td>0.028</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>99</td>
      <td>690</td>
      <td>196</td>
      <td>15</td>
      <td>0.789</td>
      <td>0.114</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>1</td>
      <td>743</td>
      <td>255</td>
      <td>1</td>
      <td>0.744</td>
      <td>0.002</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>114</td>
      <td>543</td>
      <td>265</td>
      <td>78</td>
      <td>0.657</td>
      <td>0.192</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>40</td>
      <td>789</td>
      <td>158</td>
      <td>13</td>
      <td>0.829</td>
      <td>0.053</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>28</td>
      <td>731</td>
      <td>220</td>
      <td>21</td>
      <td>0.759</td>
      <td>0.049</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>64</td>
      <td>657</td>
      <td>249</td>
      <td>30</td>
      <td>0.721</td>
      <td>0.094</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>24</td>
      <td>785</td>
      <td>183</td>
      <td>8</td>
      <td>0.809</td>
      <td>0.032</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>24</td>
      <td>713</td>
      <td>259</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.028</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>14</td>
      <td>661</td>
      <td>320</td>
      <td>5</td>
      <td>0.675</td>
      <td>0.019</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>10</td>
      <td>725</td>
      <td>261</td>
      <td>4</td>
      <td>0.735</td>
      <td>0.014</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>15</td>
      <td>767</td>
      <td>213</td>
      <td>5</td>
      <td>0.782</td>
      <td>0.02</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>36</td>
      <td>658</td>
      <td>297</td>
      <td>9</td>
      <td>0.694</td>
      <td>0.045</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



`Hernia` has a prevalence 0.002, which is the rarest among the studied conditions in our dataset.

<a name='3-4'></a>
### 3.4 Sensitivity and Specificity

<img src="images/sens_spec.png" width="30%">

Sensitivity and specificity are two of the most prominent numbers that are used to measure diagnostics tests.
- Sensitivity is the probability that our test outputs positive given that the case is actually positive.
- Specificity is the probability that the test outputs negative given that the case is actually negative. 

We can phrase this easily in terms of true positives, true negatives, false positives, and false negatives: 

$$sensitivity = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}$$

$$specificity = \frac{\text{true negatives}}{\text{true negatives} + \text{false positives}}$$

<a name='ex-4'></a>
### Exercise 4 - get_sensitivity and get_specificity

Let's calculate sensitivity and specificity for our model:


```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_sensitivity(y, pred, th=0.5):
    """
    Compute sensitivity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        sensitivity (float): probability that our test outputs positive given that the case is actually positive
    """
    sensitivity = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # get TP and FN using our previously defined functions
    TP = true_positives(y, pred, th)
    FN = false_negatives(y, pred, th)

    # use TP and FN to compute sensitivity
    sensitivity = TP / (TP + FN)
    
    ### END CODE HERE ###
    
    return sensitivity

def get_specificity(y, pred, th=0.5):
    """
    Compute specificity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        specificity (float): probability that the test outputs negative given that the case is actually negative
    """
    specificity = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # get TN and FP using our previously defined functions
    TN = true_negatives(y, pred, th)
    FP = false_positives(y, pred, th)
    
    # use TN and FP to compute specificity 
    specificity = TN / (TN + FP)
    
    ### END CODE HERE ###
    
    return specificity
```


```python
### do not modify this cell    
get_sensitivity_specificity_test(get_sensitivity, get_specificity)
```

    Test Case:
    
    Test Labels:	       [1 0 0 1 1]
    Test Predictions:      [1 0 0 1 1]
    Threshold:	       0.5
    Computed Sensitivity:  0.6666666666666666
    Computed Specificity:  0.5 
    
    [92m All tests passed.
    [92m All tests passed.


#### Expected output:

```Python
Test Case:

Test Labels:	       [1 0 0 1 1]
Test Predictions:      [1 0 0 1 1]
Threshold:	         0.5
Computed Sensitivity:  0.6666666666666666
Computed Specificity:  0.5 
```
```
 All tests passed.
 All tests passed.

```


```python
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Prevalence</th>
      <th>Sensitivity</th>
      <th>Specificity</th>
      <th>PPV</th>
      <th>NPV</th>
      <th>AUC</th>
      <th>F1</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>16</td>
      <td>814</td>
      <td>169</td>
      <td>1</td>
      <td>0.83</td>
      <td>0.017</td>
      <td>0.941</td>
      <td>0.828</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>20</td>
      <td>869</td>
      <td>103</td>
      <td>8</td>
      <td>0.889</td>
      <td>0.028</td>
      <td>0.714</td>
      <td>0.894</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>99</td>
      <td>690</td>
      <td>196</td>
      <td>15</td>
      <td>0.789</td>
      <td>0.114</td>
      <td>0.868</td>
      <td>0.779</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>1</td>
      <td>743</td>
      <td>255</td>
      <td>1</td>
      <td>0.744</td>
      <td>0.002</td>
      <td>0.5</td>
      <td>0.744</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>114</td>
      <td>543</td>
      <td>265</td>
      <td>78</td>
      <td>0.657</td>
      <td>0.192</td>
      <td>0.594</td>
      <td>0.672</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>40</td>
      <td>789</td>
      <td>158</td>
      <td>13</td>
      <td>0.829</td>
      <td>0.053</td>
      <td>0.755</td>
      <td>0.833</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>28</td>
      <td>731</td>
      <td>220</td>
      <td>21</td>
      <td>0.759</td>
      <td>0.049</td>
      <td>0.571</td>
      <td>0.769</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>64</td>
      <td>657</td>
      <td>249</td>
      <td>30</td>
      <td>0.721</td>
      <td>0.094</td>
      <td>0.681</td>
      <td>0.725</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>24</td>
      <td>785</td>
      <td>183</td>
      <td>8</td>
      <td>0.809</td>
      <td>0.032</td>
      <td>0.75</td>
      <td>0.811</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>24</td>
      <td>713</td>
      <td>259</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.028</td>
      <td>0.857</td>
      <td>0.734</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>14</td>
      <td>661</td>
      <td>320</td>
      <td>5</td>
      <td>0.675</td>
      <td>0.019</td>
      <td>0.737</td>
      <td>0.674</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>10</td>
      <td>725</td>
      <td>261</td>
      <td>4</td>
      <td>0.735</td>
      <td>0.014</td>
      <td>0.714</td>
      <td>0.735</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>15</td>
      <td>767</td>
      <td>213</td>
      <td>5</td>
      <td>0.782</td>
      <td>0.02</td>
      <td>0.75</td>
      <td>0.783</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>36</td>
      <td>658</td>
      <td>297</td>
      <td>9</td>
      <td>0.694</td>
      <td>0.045</td>
      <td>0.8</td>
      <td>0.689</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



Note that specificity and sensitivity do not depend on the prevalence of the positive class in the dataset. 
* This is because the statistics are only computed within people of the same class
* Sensitivity only considers output on people in the positive class
* Similarly, specificity only considers output on people in the negative class.

<a name='3-5'></a>
### 3.5 PPV and NPV

Diagnostically, however, sensitivity and specificity are not helpful. Sensitivity, for example, tells us the probability our test outputs positive given that the person already has the condition. Here, we are conditioning on the thing we would like to find out (whether the patient has the condition)!

What would be more helpful is the probability that the person has the disease given that our test outputs positive. That brings us to positive predictive value (PPV) and negative predictive value (NPV).

- Positive predictive value (PPV) is the probability that subjects with a positive screening test truly have the disease.
- Negative predictive value (NPV) is the probability that subjects with a negative screening test truly don't have the disease.

Again, we can formulate these in terms of true positives, true negatives, false positives, and false negatives: 

$$PPV = \frac{\text{true positives}}{\text{true positives} + \text{false positives}}$$ 

$$NPV = \frac{\text{true negatives}}{\text{true negatives} + \text{false negatives}}$$

<a name='ex-5'></a>
### Exercise 5 - get_ppv and get_npv

Let's calculate PPV & NPV for our model:


```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_ppv(y, pred, th=0.5):
    """
    Compute PPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        PPV (float): positive predictive value of predictions at threshold
    """
    PPV = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # get TP and FP using our previously defined functions
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)

    # use TP and FP to compute PPV
    PPV = TP / (TP + FP)
    
    ### END CODE HERE ###
    
    return PPV

def get_npv(y, pred, th=0.5):
    """
    Compute NPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        NPV (float): negative predictive value of predictions at threshold
    """
    NPV = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # get TN and FN using our previously defined functions
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)

    # use TN and FN to compute NPV
    NPV = TN / (TN + FN)
    
    ### END CODE HERE ###
    
    return NPV
```


```python
### do not modify this cell    
get_ppv_npv_test(get_ppv, get_npv)    
```

    Test Case:
    
    Test Labels:	   [1 0 0 1 1]
    Test Predictions:  [1 0 0 1 1]
    Threshold:	   0.5
    Computed PPV:	   0.6666666666666666
    Computed NPV:	   0.5 
    
    [92m All tests passed.
    [92m All tests passed.


#### Expected output:

```Python
Test Case:

Test Labels:	   [1 0 0 1 1]
Test Predictions:  [1 0 0 1 1]
Threshold:	     0.5
Computed PPV:	  0.6666666666666666
Computed NPV:	  0.5 
```
```
 All tests passed.
 All tests passed.
```


```python
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Prevalence</th>
      <th>Sensitivity</th>
      <th>Specificity</th>
      <th>PPV</th>
      <th>NPV</th>
      <th>AUC</th>
      <th>F1</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>16</td>
      <td>814</td>
      <td>169</td>
      <td>1</td>
      <td>0.83</td>
      <td>0.017</td>
      <td>0.941</td>
      <td>0.828</td>
      <td>0.086</td>
      <td>0.999</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>20</td>
      <td>869</td>
      <td>103</td>
      <td>8</td>
      <td>0.889</td>
      <td>0.028</td>
      <td>0.714</td>
      <td>0.894</td>
      <td>0.163</td>
      <td>0.991</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>99</td>
      <td>690</td>
      <td>196</td>
      <td>15</td>
      <td>0.789</td>
      <td>0.114</td>
      <td>0.868</td>
      <td>0.779</td>
      <td>0.336</td>
      <td>0.979</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>1</td>
      <td>743</td>
      <td>255</td>
      <td>1</td>
      <td>0.744</td>
      <td>0.002</td>
      <td>0.5</td>
      <td>0.744</td>
      <td>0.004</td>
      <td>0.999</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>114</td>
      <td>543</td>
      <td>265</td>
      <td>78</td>
      <td>0.657</td>
      <td>0.192</td>
      <td>0.594</td>
      <td>0.672</td>
      <td>0.301</td>
      <td>0.874</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>40</td>
      <td>789</td>
      <td>158</td>
      <td>13</td>
      <td>0.829</td>
      <td>0.053</td>
      <td>0.755</td>
      <td>0.833</td>
      <td>0.202</td>
      <td>0.984</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>28</td>
      <td>731</td>
      <td>220</td>
      <td>21</td>
      <td>0.759</td>
      <td>0.049</td>
      <td>0.571</td>
      <td>0.769</td>
      <td>0.113</td>
      <td>0.972</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>64</td>
      <td>657</td>
      <td>249</td>
      <td>30</td>
      <td>0.721</td>
      <td>0.094</td>
      <td>0.681</td>
      <td>0.725</td>
      <td>0.204</td>
      <td>0.956</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>24</td>
      <td>785</td>
      <td>183</td>
      <td>8</td>
      <td>0.809</td>
      <td>0.032</td>
      <td>0.75</td>
      <td>0.811</td>
      <td>0.116</td>
      <td>0.99</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>24</td>
      <td>713</td>
      <td>259</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.028</td>
      <td>0.857</td>
      <td>0.734</td>
      <td>0.085</td>
      <td>0.994</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>14</td>
      <td>661</td>
      <td>320</td>
      <td>5</td>
      <td>0.675</td>
      <td>0.019</td>
      <td>0.737</td>
      <td>0.674</td>
      <td>0.042</td>
      <td>0.992</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>10</td>
      <td>725</td>
      <td>261</td>
      <td>4</td>
      <td>0.735</td>
      <td>0.014</td>
      <td>0.714</td>
      <td>0.735</td>
      <td>0.037</td>
      <td>0.995</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>15</td>
      <td>767</td>
      <td>213</td>
      <td>5</td>
      <td>0.782</td>
      <td>0.02</td>
      <td>0.75</td>
      <td>0.783</td>
      <td>0.066</td>
      <td>0.994</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>36</td>
      <td>658</td>
      <td>297</td>
      <td>9</td>
      <td>0.694</td>
      <td>0.045</td>
      <td>0.8</td>
      <td>0.689</td>
      <td>0.108</td>
      <td>0.987</td>
      <td>Not Defined</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



Notice that despite having very high sensitivity and accuracy, the PPV of the predictions could still be very low. 

This is the case with `Edema`, for example. 
* The sensitivity for `Edema` is 0.75.
* However, given that the model predicted positive, the probability that a person has Edema (its PPV) is only 0.066!

<a name='3-6'></a>
### 3.6 ROC Curve

So far we have been operating under the assumption that our model's prediction of `0.5` and above should be treated as positive and otherwise it should be treated as negative. This however was a rather arbitrary choice. One way to see this, is to look at a very informative visualization called the receiver operating characteristic (ROC) curve.

The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The ideal point is at the top left, with a true positive rate of 1 and a false positive rate of 0. The various points on the curve are generated by gradually changing the threshold.

Let's look at this curve for our model:


```python
util.get_curve(y, pred, class_labels)
```


![png](output_51_0.png)


The area under the ROC curve is also called AUCROC or C-statistic and is a measure of goodness of fit. In medical literature this number also gives the probability that a randomly selected patient who experienced a condition had a higher risk score than a patient who had not experienced the event. This summarizes the model output across all thresholds, and provides a good sense of the discriminative power of a given model.

Let's use the `sklearn` metric function of `roc_auc_score` to add this score to our metrics table.


```python
from sklearn.metrics import roc_auc_score
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Prevalence</th>
      <th>Sensitivity</th>
      <th>Specificity</th>
      <th>PPV</th>
      <th>NPV</th>
      <th>AUC</th>
      <th>F1</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>16</td>
      <td>814</td>
      <td>169</td>
      <td>1</td>
      <td>0.83</td>
      <td>0.017</td>
      <td>0.941</td>
      <td>0.828</td>
      <td>0.086</td>
      <td>0.999</td>
      <td>0.933</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>20</td>
      <td>869</td>
      <td>103</td>
      <td>8</td>
      <td>0.889</td>
      <td>0.028</td>
      <td>0.714</td>
      <td>0.894</td>
      <td>0.163</td>
      <td>0.991</td>
      <td>0.935</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>99</td>
      <td>690</td>
      <td>196</td>
      <td>15</td>
      <td>0.789</td>
      <td>0.114</td>
      <td>0.868</td>
      <td>0.779</td>
      <td>0.336</td>
      <td>0.979</td>
      <td>0.891</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>1</td>
      <td>743</td>
      <td>255</td>
      <td>1</td>
      <td>0.744</td>
      <td>0.002</td>
      <td>0.5</td>
      <td>0.744</td>
      <td>0.004</td>
      <td>0.999</td>
      <td>0.644</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>114</td>
      <td>543</td>
      <td>265</td>
      <td>78</td>
      <td>0.657</td>
      <td>0.192</td>
      <td>0.594</td>
      <td>0.672</td>
      <td>0.301</td>
      <td>0.874</td>
      <td>0.696</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>40</td>
      <td>789</td>
      <td>158</td>
      <td>13</td>
      <td>0.829</td>
      <td>0.053</td>
      <td>0.755</td>
      <td>0.833</td>
      <td>0.202</td>
      <td>0.984</td>
      <td>0.888</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>28</td>
      <td>731</td>
      <td>220</td>
      <td>21</td>
      <td>0.759</td>
      <td>0.049</td>
      <td>0.571</td>
      <td>0.769</td>
      <td>0.113</td>
      <td>0.972</td>
      <td>0.745</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>64</td>
      <td>657</td>
      <td>249</td>
      <td>30</td>
      <td>0.721</td>
      <td>0.094</td>
      <td>0.681</td>
      <td>0.725</td>
      <td>0.204</td>
      <td>0.956</td>
      <td>0.781</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>24</td>
      <td>785</td>
      <td>183</td>
      <td>8</td>
      <td>0.809</td>
      <td>0.032</td>
      <td>0.75</td>
      <td>0.811</td>
      <td>0.116</td>
      <td>0.99</td>
      <td>0.826</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>24</td>
      <td>713</td>
      <td>259</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.028</td>
      <td>0.857</td>
      <td>0.734</td>
      <td>0.085</td>
      <td>0.994</td>
      <td>0.868</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>14</td>
      <td>661</td>
      <td>320</td>
      <td>5</td>
      <td>0.675</td>
      <td>0.019</td>
      <td>0.737</td>
      <td>0.674</td>
      <td>0.042</td>
      <td>0.992</td>
      <td>0.762</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>10</td>
      <td>725</td>
      <td>261</td>
      <td>4</td>
      <td>0.735</td>
      <td>0.014</td>
      <td>0.714</td>
      <td>0.735</td>
      <td>0.037</td>
      <td>0.995</td>
      <td>0.801</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>15</td>
      <td>767</td>
      <td>213</td>
      <td>5</td>
      <td>0.782</td>
      <td>0.02</td>
      <td>0.75</td>
      <td>0.783</td>
      <td>0.066</td>
      <td>0.994</td>
      <td>0.856</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>36</td>
      <td>658</td>
      <td>297</td>
      <td>9</td>
      <td>0.694</td>
      <td>0.045</td>
      <td>0.8</td>
      <td>0.689</td>
      <td>0.108</td>
      <td>0.987</td>
      <td>0.799</td>
      <td>Not Defined</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



<a name='4'></a>
## 4. Confidence Intervals

Of course our dataset is only a sample of the real world, and our calculated values for all above metrics is an estimate of the real world values. It would be good to quantify this uncertainty due to the sampling of our dataset. We'll do this through the use of confidence intervals. A 95\% confidence interval for an estimate $\hat{s}$ of a parameter $s$ is an interval $I = (a, b)$ such that 95\% of the time when the experiment is run, the true value $s$ is contained in $I$. More concretely, if we were to run the experiment many times, then the fraction of those experiments for which $I$ contains the true parameter would tend towards 95\%.

While some estimates come with methods for computing the confidence interval analytically, more complicated statistics, such as the AUC for example, are difficult. For these we can use a method called the *bootstrap*. The bootstrap estimates the uncertainty by resampling the dataset with replacement. For each resampling $i$, we will get a new estimate, $\hat{s}_i$. We can then estimate the distribution of $\hat{s}$ by using the distribution of $\hat{s}_i$ for our bootstrap samples.

In the code below, we create bootstrap samples and compute sample AUCs from those samples. Note that we use stratified random sampling (sampling from the positive and negative classes separately) to make sure that members of each class are represented. 


```python
def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

statistics = bootstrap_auc(y, pred, class_labels)
```

Now we can compute confidence intervals from the sample statistics that we computed.


```python
util.print_confidence_intervals(class_labels, statistics)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean AUC (CI 5%-95%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>0.93 (0.89-0.96)</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>0.94 (0.91-0.96)</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>0.89 (0.87-0.91)</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>0.61 (0.30-0.98)</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>0.70 (0.67-0.72)</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>0.88 (0.83-0.92)</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>0.75 (0.67-0.81)</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>0.78 (0.74-0.82)</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>0.82 (0.76-0.88)</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>0.86 (0.79-0.91)</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>0.76 (0.68-0.85)</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>0.80 (0.75-0.86)</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.86 (0.81-0.90)</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>0.80 (0.73-0.85)</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, our confidence intervals are much wider for some classes than for others. Hernia, for example, has an interval around (0.30 - 0.98), indicating that we can't be certain it is better than chance (at 0.5). 

<a name='5'></a>
## 5. Precision-Recall Curve


Precision-Recall are informative prediction metrics when significant class imbalance are present in the data.

In information retrieval
- Precision is a measure of result relevancy and that is equivalent to our previously defined PPV. 
- Recall is a measure of how many truly relevant results are returned and that is equivalent to our previously defined sensitivity measure.

The precision-recall curve (PRC) shows the trade-off between precision and recall for different thresholds. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. 

High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).

Run the following cell to generate a PRC:


```python
util.get_curve(y, pred, class_labels, curve='prc')
```


![png](output_62_0.png)


<a name='6'></a>
## 6. F1 Score

F1 score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. 

Again, we can simply use `sklearn`'s utility metric function of `f1_score` to add this measure to our performance table.


```python
from sklearn.metrics import f1_score
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score,f1=f1_score)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Prevalence</th>
      <th>Sensitivity</th>
      <th>Specificity</th>
      <th>PPV</th>
      <th>NPV</th>
      <th>AUC</th>
      <th>F1</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiomegaly</th>
      <td>16</td>
      <td>814</td>
      <td>169</td>
      <td>1</td>
      <td>0.83</td>
      <td>0.017</td>
      <td>0.941</td>
      <td>0.828</td>
      <td>0.086</td>
      <td>0.999</td>
      <td>0.933</td>
      <td>0.158</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>20</td>
      <td>869</td>
      <td>103</td>
      <td>8</td>
      <td>0.889</td>
      <td>0.028</td>
      <td>0.714</td>
      <td>0.894</td>
      <td>0.163</td>
      <td>0.991</td>
      <td>0.935</td>
      <td>0.265</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>99</td>
      <td>690</td>
      <td>196</td>
      <td>15</td>
      <td>0.789</td>
      <td>0.114</td>
      <td>0.868</td>
      <td>0.779</td>
      <td>0.336</td>
      <td>0.979</td>
      <td>0.891</td>
      <td>0.484</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>1</td>
      <td>743</td>
      <td>255</td>
      <td>1</td>
      <td>0.744</td>
      <td>0.002</td>
      <td>0.5</td>
      <td>0.744</td>
      <td>0.004</td>
      <td>0.999</td>
      <td>0.644</td>
      <td>0.008</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>114</td>
      <td>543</td>
      <td>265</td>
      <td>78</td>
      <td>0.657</td>
      <td>0.192</td>
      <td>0.594</td>
      <td>0.672</td>
      <td>0.301</td>
      <td>0.874</td>
      <td>0.696</td>
      <td>0.399</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>40</td>
      <td>789</td>
      <td>158</td>
      <td>13</td>
      <td>0.829</td>
      <td>0.053</td>
      <td>0.755</td>
      <td>0.833</td>
      <td>0.202</td>
      <td>0.984</td>
      <td>0.888</td>
      <td>0.319</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>28</td>
      <td>731</td>
      <td>220</td>
      <td>21</td>
      <td>0.759</td>
      <td>0.049</td>
      <td>0.571</td>
      <td>0.769</td>
      <td>0.113</td>
      <td>0.972</td>
      <td>0.745</td>
      <td>0.189</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Atelectasis</th>
      <td>64</td>
      <td>657</td>
      <td>249</td>
      <td>30</td>
      <td>0.721</td>
      <td>0.094</td>
      <td>0.681</td>
      <td>0.725</td>
      <td>0.204</td>
      <td>0.956</td>
      <td>0.781</td>
      <td>0.314</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>24</td>
      <td>785</td>
      <td>183</td>
      <td>8</td>
      <td>0.809</td>
      <td>0.032</td>
      <td>0.75</td>
      <td>0.811</td>
      <td>0.116</td>
      <td>0.99</td>
      <td>0.826</td>
      <td>0.201</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>24</td>
      <td>713</td>
      <td>259</td>
      <td>4</td>
      <td>0.737</td>
      <td>0.028</td>
      <td>0.857</td>
      <td>0.734</td>
      <td>0.085</td>
      <td>0.994</td>
      <td>0.868</td>
      <td>0.154</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>14</td>
      <td>661</td>
      <td>320</td>
      <td>5</td>
      <td>0.675</td>
      <td>0.019</td>
      <td>0.737</td>
      <td>0.674</td>
      <td>0.042</td>
      <td>0.992</td>
      <td>0.762</td>
      <td>0.079</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>10</td>
      <td>725</td>
      <td>261</td>
      <td>4</td>
      <td>0.735</td>
      <td>0.014</td>
      <td>0.714</td>
      <td>0.735</td>
      <td>0.037</td>
      <td>0.995</td>
      <td>0.801</td>
      <td>0.07</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>15</td>
      <td>767</td>
      <td>213</td>
      <td>5</td>
      <td>0.782</td>
      <td>0.02</td>
      <td>0.75</td>
      <td>0.783</td>
      <td>0.066</td>
      <td>0.994</td>
      <td>0.856</td>
      <td>0.121</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>36</td>
      <td>658</td>
      <td>297</td>
      <td>9</td>
      <td>0.694</td>
      <td>0.045</td>
      <td>0.8</td>
      <td>0.689</td>
      <td>0.108</td>
      <td>0.987</td>
      <td>0.799</td>
      <td>0.19</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



<a name='7'></a>
## 7. Calibration

When performing classification we often want not only to predict the class label, but also obtain a probability of each label. This probability would ideally give us some kind of confidence on the prediction. In order to observe how our model's generated probabilities are aligned with the real probabilities, we can plot what's called a *calibration curve*. 

In order to generate a calibration plot, we first bucketize our predictions to a fixed number of separate bins (e.g. 5) between 0 and 1. We then calculate a point for each bin: the x-value for each point is the mean for the probability that our model has assigned to these points and the y-value for each point fraction of true positives in that bin. We then plot these points in a linear plot. A well-calibrated model has a calibration curve that almost aligns with the y=x line.

The `sklearn` library has a utility `calibration_curve` for generating a calibration plot. Let's use it and take a look at our model's calibration:


```python
from sklearn.calibration import calibration_curve
def plot_calibration_curve(y, pred):
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()
```


```python
plot_calibration_curve(y, pred)
```


![png](output_69_0.png)


As the above plots show, for most predictions our model's calibration plot does not resemble a well calibrated plot. How can we fix that?...

Thankfully, there is a very useful method called [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) which works by fitting a logistic regression model to our model's scores. To build this model, we will be using the training portion of our dataset to generate the linear model and then will use the model to calibrate the predictions for our test portion.


```python
from sklearn.linear_model import LogisticRegression as LR 

y_train = train_results[class_labels].values
pred_train = train_results[pred_labels].values
pred_calibrated = np.zeros_like(pred)

for i in range(len(class_labels)):
    lr = LR(solver='liblinear', max_iter=10000)
    lr.fit(pred_train[:, i].reshape(-1, 1), y_train[:, i])    
    pred_calibrated[:, i] = lr.predict_proba(pred[:, i].reshape(-1, 1))[:,1]
```


```python
plot_calibration_curve(y[:,], pred_calibrated)
```


![png](output_72_0.png)


# That's it!
Congratulations! That was a lot of metrics to get familiarized with. 
We hope that you feel a lot more confident in your understanding of medical diagnostic evaluation and test your models correctly in your future work :)
