# Machine Learning for Regression

## Car Price Prediction

This project is about the creation of a model for helping users to predict car prices. The dataset was obtained from [this 
kaggle competition](https://www.kaggle.com/CooperUnion/cardataset).

**Project plan:**

* Prepare data and Exploratory data analysis (EDA)
* Use linear regression for predicting price
* Understanding the internals of linear regression 
* Evaluating the model with RMSE
* Feature engineering  
* Regularization 
* Using the model

### Pandas attributes and methods

* pd.read_csv(<file_path_string>) - read csv files 
* df.head() - take a look of the dataframe 
* df.columns - retrieve colum names of a dataframe 
* df.columns.str.lower() - lowercase all the letters 
* df.columns.str.replace(' ', '_') - replace the space separator 
* df.dtypes - retrieve data types of all features 
* df.index - retrieve indices of a dataframe

### Exploratory data analysis (EDA)

**Pandas attributes and methods:** 

* df[col].unique() - returns a list of unique values in the series 
* df[col].nunique() - returns the number of unique values in the series 
* df.isnull().sum() - returns the number of null values in the dataframe 

**Matplotlib and seaborn methods:**

* %matplotlib inline - assure that plots are displayed in jupyter notebook's cells
* sns.histplot() - show the histogram of a series 
   
**Numpy methods:**
* np.log1p() - applies log transformation to a variable and adds one to each result.

Long-tail distributions usually confuse the ML models, so the recommendation is to transform the target variable distribution to a normal one whenever possible. 


### Validation Framework

In general, the dataset is split into three parts: training, validation, and test. For each partition, we need to obtain feature matrices (X) and y vectors of targets. First, the size of partitions is calculated, records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices. 

**Pandas attributes and methods:** 

* `df.iloc[]` - returns subsets of records of a dataframe, being selected by numerical indices
* `df.reset_index()` - restate the orginal indices 
* `del df[col]` - eliminates target variable 

**Numpy methods:**

* `np.arange()` - returns an array of numbers 
* `np.random.shuffle()` - returns a shuffled array
* `np.random.seed()` - set a seed 

### Linear Regression

Model for solving regression tasks, in which the objective is to adjust a line for the data and make predictions on new values. The input of this model is the **feature matrix** `X` and a `y` **vector of predictions** is obtained, trying to be as close as possible to the **actual** `y` values. The linear regression formula is the sum of the bias term \( $w_0$ \), which refers to the predictions if there is no information, and each of the feature values times their corresponding weights as \( $x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$ \).

So the simple linear regression formula looks like:

$g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.

And that can be further simplified as:

$g(x_i) = w_0 + \displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$

Here is a simple implementation of Linear Regression in python:

~~~~python
w0 = 7.1
def linear_regression(xi):
    
    n = len(xi)
    
    pred = w0
    w = [0.01, 0.04, 0.002]
    for j in range(n):
        pred = pred + w[j] * xi[j]
    return pred
~~~~
        

If we look at the $\displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$ part in the above equation, we know that this is nothing else but a vector-vector multiplication. Hence, we can rewrite the equation as $g(x_i) = w_0 + x_i^T \cdot w$

We need to assure that the result is shown on the untransformed scale by using the inverse function `exp()`. 

### Linear Regression: vector form

The formula of linear regression can be synthesized with the dot product between features and weights. The feature vector includes the *bias* term with an *x* value of one, such as $w_{0}^{x_{i0}},\ where\ x_{i0} = 1\ for\ w_0$.

When all the records are included, the linear regression can be calculated with the dot product between ***feature matrix*** and ***vector of weights***, obtaining the `y` vector of predictions. 

### Linear Regression: training

Obtaining predictions as close as possible to $y$ target values requires the calculation of weights from the general
LR equation. The feature matrix does not 
have an inverse because it is not square, so it is required to obtain an approximate solution, which can be
obtained using the **Gram matrix** 
(multiplication of feature matrix ($X$) and its transpose ($X^T$)). The vector of weights or coefficients $w$ obtained with this
formula is the closest possible solution to the LR system.

Normal Equation:

$w$ = $(X^TX)^{-1}X^Ty$

Where:

$X^TX$ is the Gram Matrix

### Baseline Model

* In this lesson we build a baseline model and apply the `df_train` dataset to derive weights for the bias (w0) and the features (w). For this, we use the `train_linear_regression(X, y)` function from the previous lesson.
* Linear regression only applies to numerical features. Therefore, only the numerical features from `df_train` are used for the feature matrix. 
* We notice some of the features in `df_train` are `nan`. We set them to `0` for the sake of simplicity, so the model is solvable, but it will be appropriate if a non-zeo value is used as the filler (e.g. mean value of the feature).
* Once the weights are calculated, then we apply them on  $$\\\\ \large g(X) = w_0 + X \cdot w$$ to derive the predicted y vector.
* Then we plot both predicted y and the actual y on the same histogram for a visual comparison.

### Root Mean Squared Error (RMSE)

* In the previous lesson we found out our predictions were a bit off from the actual target values in the training dataset. We need a way to quantify how good or bad the model is. This is where RMSE can be of help.
* Root Mean Squared Error (RMSE) is a way to evaluate regression models. It measures the error associated with the model being evaluated. This numerical figure then can be used to compare the models, enabling us to choose the one that gives the best predictions.

$$RMSE = \sqrt{ \frac{1}{m} \sum {(g(x_i) - y_i)^2}}$$

- $g(x_i)$ is the prediction
- $y_i$ is the actual
- $m$ is the number of observations in the dataset (i.e. cars)

Calculation of the RMSE on validation partition of the dataset of car price prediction. In this way, we have a metric to evaluate the model's 
performance. 

### Feature Engineering

The feature age of the car was included in the dataset, obtained with the subtraction of the maximum year of cars and each of the years of cars. 
This new feature improved the model performance, measured with the RMSE and comparing the distributions of y target variable and predictions. 

Categorical variables are typically strings, and pandas identifies them as object types. These variables need to be converted to a numerical form because ML models can interpret only numerical features. It is possible to incorporate certain categories from a feature, not necessarily all of them. 
This transformation from categorical to numerical variables is known as One-Hot encoding. 

### Regularization

If the feature matrix has duplicate columns (or columns that can be expressed as a linear combination of other columns), it will not have an inverse matrix. But, sometimes this error could be passed if certain values are slightly different between duplicated columns. 

So, if we apply the normal equation with this feature matrix, the values associated with duplicated columns are very large, which decreases the model performance. To solve this issue, one alternative is adding a small number to the diagonal of the feature matrix, which corresponds to regularization. 

This technique works because the addition of small values to the diagonal makes it less likely to have duplicated columns. The regularization value is a hyperparameter of the model. After applying regularization the model performance improved.

### Tuning the model

Tuning the model consisted of finding the best regularization hyperparameter value, using the validation partition of the dataset. The model was then trained with this regularization value. 

### Using the model

After finding the best model and its parameters, it was trained with training and validation partitions and the final RMSE was calculated on the test partition. 

Finally, the final model was used to predict the price of new cars. 