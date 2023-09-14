# Introduction to Machine Learning

## What is Machine Learning?

The concept of ML is depicted with an example of predicting the price of a car. The ML model
learns from data, represented as some **features** such as year, mileage, among others, and the **target** variable, in this
case, the car's price, by extracting patterns from the data.

Then, the model is given new data (**without** the target) about cars and predicts their price (target). 

In summary, ML is a process of **extracting patterns from data**, which is of two types:

* features (information about the object) and 
* target (property to predict for unseen objects). 

Therefore, new feature values are presented to the model, and it makes **predictions** from the learned patterns.

## Machine Learning vs Rule-Based Systems

The differences between ML and Rule-Based systems is explained with the example of a **spam filter**.

Traditional Rule-Based systems are based on a set of **characteristics** (keywords, email length, etc.) that identify an email as spam or not. As spam emails keep changing over time the system needs to be upgraded making the process untractable due to the complexity of code maintenance as the system grows.

ML can be used to solve this problem with the following steps:

### Get data 
Emails from the user's spam folder and inbox gives examples of spam and non-spam.

### Define and calculate features
Rules/characteristics from rule-based systems can be used as a starting point to define features for the ML model. The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox).

Each email can be encoded (converted) to the values of it's features and target.

### Train and use the model
A machine learning algorithm can then be applied to the encoded emails to build a model that can predict whether a new email is spam or not spam. The **predictions are probabilities**, and to make a decision it is necessary to define a threshold to classify emails as spam or not spam.

## Supervised Machine Learning

In Supervised Machine Learning (SML) there are always labels associated with certain features.
The model is trained, and then it can make predictions on new features. In this way, the model is taught by certain features and targets. 

* **Feature matrix (X):** made of observations or objects (rows) and features (columns).
* **Target variable (y):** a vector with the target information we want to predict. For each row of X there's a value in y.


The model can be represented as a function **g** that takes the X matrix as a parameter and tries to predict values as close as possible to y targets. 
The obtention of the g function is what it is called **training**.

### Types of SML problems 

* **Regression:** the output is a number (car's price)
* **Classification:** the output is a category (spam example). 
	* **Binary:** there are two categories. 
	* **Multiclass problems:** there are more than two categories. 
* **Ranking:** the output is the big scores associated with certain items. It is applied in recommender systems. 

In summary, SML is about teaching the model by showing different examples, and the goal is to come up with a function that takes the feature matrix as a
parameter and makes predictions as close as possible to the y targets.

## CRISP - DM

CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an open standard process model that describes common approaches used by data mining experts. It is the most widely-used analytics model. Was conceived in 1996 and became a European Union project under the ESPRIT funding initiative in 1997. The project was led by five companies: Integral Solutions Ltd (ISL), Teradata, Daimler AG, NCR Corporation and OHRA, an insurance company: 

1. **Business understanding:** An important question is if: **do we need ML for the project**. The goal of the project has to be measurable, i.e. reduce the amount of spam by 50%.
2. **Data understanding:** Analyze available data sources, and decide if more data is required.
    - Is available data reliable?
    - Do we track data correctly?
    - Is the dataset large enough?
    - Do we have enough features?
    - Do we need to get more data?
3. **Data preparation:** Clean data and remove noise applying pipelines, and the data should be converted to a tabular format, so we can put it into ML.
4. **Modeling:** training Different models and choose the best one. Considering the results of this step, it is proper to decide if is required to add new features or fix data issues. 
5. **Evaluation:** Measure how well the model is performing and if it solves the business problem.
- Is the model good enough?:
    - Have we reached the business goal?
    - Do our metrics improve and by how much?
- Do a retrospective:
    - Was the goal achievable?
    - Did we solve/measure the right thing?
- After that, we may decide to:
    - Go back and adjust the goal
    - Roll the model to more users/all users
    - Stop working on the project
6. **Deployment:** Roll out to production to all the users. The evaluation and deployment often happen together - **online evaluation** (evaluation on live users). It means: deploy the model, evaluate it. First evaluate on a small percentage of users (5%), and then roll out to all users.
- Proper monitoring
- Ensuring the quality and maintanability of the project

It is important to consider how well maintainable the project is.
  
In general, ML projects require many iterations.

**Iteration:** 
* Start simple
* Learn from the feedback
* Improve

## Model selection

### Which model to choose?

- Logistic regression
- Decision tree
- Neural Network
- Or many others

The validation dataset is not used in training. There are feature matrices and y vectors for both training and validation datasets. 
The model is fitted with training data, and it is used to predict the y values of the validation feature matrix. Then, the predicted y values (probabilities) are compared with the actual y values. 

**Multiple comparisons problem (MCP):** just by chance one model can be lucky and obtain good predictions because all of them are probabilistic. 

The test set can help to avoid the MCP. Obtaining the best model is done with the training and validation datasets, while the test dataset is used for assuring that the proposed best model is the best. 

1. Split datasets in training, validation, and test. E.g. 60%, 20% and 20% respectively 
2. Train the models
3. Evaluate the models
4. Select the best model 
5. Apply the best model to the test dataset 
6. Compare the performance metrics of validation and test

**Training Dataset:**

- Definition: The training dataset is the portion of the dataset used to train or teach a machine learning model. It contains a labeled set of examples or data points, where each data point consists of input features and their corresponding target labels or outcomes.
- Purpose: The primary purpose of the training dataset is to allow the machine learning model to learn patterns, relationships, and associations in the data. The model uses this data to adjust its internal parameters and develop a predictive or classification model.

**Validation Dataset:**

- Definition: The validation dataset is a separate dataset that is not used for training the model. Like the training dataset, it contains labeled examples, but it is reserved for model evaluation during the training process.
- Purpose: The validation dataset is used to fine-tune the model's hyperparameters (e.g., learning rate, model complexity) and monitor its performance during training. It helps in preventing overfitting, where a model becomes too specialized to the training data and performs poorly on new, unseen data.

**Testing Dataset:**

- Definition: The testing dataset is also distinct from the training and validation datasets. It contains unlabeled examples with input features but without corresponding target labels.
- Purpose: The testing dataset is used to assess the model's generalization performance. After training and fine-tuning using the training and validation datasets, the model is tested on the testing dataset to evaluate how well it can make predictions on new, unseen data. This evaluation provides insights into the model's overall effectiveness and potential for real-world use.

## Linear Algebra Refresher

* Vector operations
* Multiplication
  * Vector-vector multiplication
  * Matrix-vector multiplication
  * Matrix-matrix multiplication
* Identity matrix
* Inverse

### Vector operations
~~~~python
u = np.array([2, 7, 5, 6])
v = np.array([3, 4, 8, 6])

# addition 
u + v

# subtraction 
u - v

# scalar multiplication 
2 * v
~~~~
### Multiplication

#####  Vector-vector multiplication

~~~~python
def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]
    
    n = u.shape[0]
    
    result = 0.0

    for i in range(n):
        result = result + u[i] * v[i]
    
    return result
~~~~

#####  Matrix-vector multiplication

~~~~python
def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]
    
    num_rows = U.shape[0]
    
    result = np.zeros(num_rows)
    
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)
    
    return result
~~~~

#####  Matrix-matrix multiplication

~~~~python
def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0]
    
    num_rows = U.shape[0]
    num_cols = V.shape[1]
    
    result = np.zeros((num_rows, num_cols))
    
    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi
    
    return result
~~~~
### Identity matrix

~~~~python
I = np.eye(3)
~~~~
### Inverse
~~~~python
V = np.array([
    [1, 1, 2],
    [0, 0.5, 1], 
    [0, 2, 1],
])
inv = np.linalg.inv(V)
~~~~

