## Deployment

In This session we talked about the earlier model we made in chapter 3 for churn prediction. <br>
This chapter containes the deployment of the model. If we want to use the model to predict new values without running the code, There's a way to do this. The way to use the model in different machines without running the code, is to deploy the model in a server (run the code and make the model). After deploying the code in a machine used as server we can make some endpoints (using api's) to connect from another machine to the server and predict values.

To deploy the model in a server there are some steps:
- After training the model save it, to use it for making predictions in future (session 02-pickle).
- Make the API endpoints in order to request predictions. (session 03-flask-intro and 04-flask-deployment)
- Some other server deployment options (sessions 5 to 9)

### Pickle

**In this session we'll cover the idea "How to use the model in future without training and evaluating the code"**
- To save the model we made before there is an option using the pickle library:
  - First install the library with the command ```pip install pickle-mixin``` if you don't have it.
  - After training the model and being the model ready for prediction process use this code to save the model for later.
  - ```python
    import pickle
    
    with open('model.bin', 'wb') as f_out: # 'wb' means write-binary
        pickle.dump((dict_vectorizer, model), f_out)
    ```
  - In the code above we'll making a binary file named model.bin and writing the dict_vectorizer for one hot encoding and model as array in it. (We will save it as binary in case it wouldn't be readable by humans)
  - To be able to use the model in future without running the code, We need to open the binary file we saved before.
  - ```python
    import pickle
    
    with open('mode.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
        dict_vectorizer, model = pickle.load(f_in)
    ## Note: never open a binary file you do not trust the source!
    ```
   - With unpacking the model and the dict_vectorizer, We're able to again predict for new input values without training a new model by re-running the code.

### Flask

#### Introduction

In this session we talked about what is a web service and how to create a simple web service.
- What is actually a web service
  - A web service is a method used to communicate between electronic devices.
  - There are some methods in web services we can use it to satisfy our problems. Here below we would list some.
    - **GET:**  GET is a method used to retrieve files, For example when we are searching for a cat image in google we are actually requesting cat images with GET method.
    - **POST:** POST is the second common method used in web services. For example in a sign up process, when we are submiting our name, username, passwords, etc we are posting our data to a server that is using the web service. (Note that there is no specification where the data goes)
    -  **PUT:** PUT is same as POST but we are specifying where the data is going to.
    -  **DELETE:** DELETE is a method that is used to request to delete some data from the server.
    -  For more information just google the HTTP methods, You'll find useful information about this.
- To create a simple web service, there are plenty libraries available in every language. Here we would like to introduce Flask library in python.
  - If you haven't installed the library just try installing it with the code ```pip install Flask```
  - To create a simple web service just run the code below:
  - ```python
    from flask import Flask
    
    app = Flask('ping') # give an identity to your web service
    
    @app.route('/ping', methods=['GET']) # use decorator to add Flask's functionality to our function
    def ping():
        return 'PONG'
    
    if __name__ == '__main__':
       app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696
    ```
   - With the code above we made a simple web server and created a route named ping that would send pong string.
   - To test it just open your browser and search ```localhost:9696/ping```, You'll see that the 'PONG' string is received. Congrats You've made a simple web server 🥳.
- To use our web server to predict new values we must modify it. See how in the next session.


#### Deployment

In this session we talked about implementing the functionality of prediction to our churn web service and how to make it usable in development environment.
- To make the web service predict the churn value for each customer we must modify the code in session 3 with the code we had in previous chapters. Below we can see how the code works in order to predict the churn value.
- In order to predict we need to first load the previous saved model and use a prediction function in a special route.
  - To load the previous saved model we use the code below:
  - ```python
    import pickle
    
    with open('churn-model.bin', 'rb') as f_in:
      dv, model = pickle.load(f_in)
    ```
  - As we had earlier to predict a value for a customer we need a function like below:
  - ```python
    def predict_single(customer, dv, model):
      X = dv.transform([customer])  ## apply the one-hot encoding feature to the customer data 
      y_pred = model.predict_proba(X)[:, 1]
      return y_pred[0]
    ```
   - Then at last we make the final function used for creating the web service.
   - ```python
     @app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
     def predict():
     customer = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.

     prediction = predict_single(customer, dv, model)
     churn = prediction >= 0.5
     
     result = {
         'churn_probability': float(prediction), ## we need to cast numpy float type to python native float type
         'churn': bool(churn),  ## same as the line above, casting the value using bool method
     }

     return jsonify(result)  ## send back the data in json format to the user
     ```
   - The whole code above is available in this link: [churn_serving.py](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-05-deployment/churn_serving.py)
   - At last run your code. To see the result we can't use a simple request in web browser, because we are expecting a `POST` request in our app. We can run the code below to **post** customer data as `json` and see the response
   - ```python     
     ## a new customer informations
     customer = {
       'customerid': '8879-zkjof',
       'gender': 'female',
       'seniorcitizen': 0,
       'partner': 'no',
       'dependents': 'no',
       'tenure': 41,
       'phoneservice': 'yes',
       'multiplelines': 'no',
       'internetservice': 'dsl',
       'onlinesecurity': 'yes',
       'onlinebackup': 'no',
       'deviceprotection': 'yes',
       'techsupport': 'yes',
       'streamingtv': 'yes',
       'streamingmovies': 'yes',
       'contract': 'one_year',
       'paperlessbilling': 'yes',
       'paymentmethod': 'bank_transfer_(automatic)',
       'monthlycharges': 79.85,
       'totalcharges': 3320.75
     }
     import requests ## to use the POST method we use a library named requests
     url = 'http://localhost:9696/predict' ## this is the route we made for prediction
     response = requests.post(url, json=customer) ## post the customer information in json format
     result = response.json() ## get the server response
     print(result)
     ```
 - Until here we saw how we made a simple web server that predicts the churn value for every user. When you run your app you will see a warning that it is not a WGSI server and not suitable for production environmnets. To fix this issue and run this as a production server there are plenty of ways available. 
   - One way to create a WSGI server is to use gunicorn. To install it use the command ```pip install gunicorn```, And to run the WGSI server you can simply run it with the   command ```gunicorn --bind 0.0.0.0:9696 churn:app```. Note that in __churn:app__ the name churn is the name we set for our the file containing the code ```app = Flask('churn')```(for example: churn.py), You may need to change it to whatever you named your Flask app file.  
   -  Windows users may not be able to use gunicorn library because windows system do not support some dependecies of the library. So to be able to run this on a windows   machine, there is an alternative library waitress and to install it just use the command ```pip install waitress```. 
   -  to run the waitress wgsi server use the command ```waitress-serve --listen=0.0.0.0:9696 churn:app```.
   -  To test it just you can run the code above and the results is the same.
 - So until here you were able to make a production server that predict the churn value for new customers. In the next session we can see how to solve library version conflictions in each machine and manage the dependencies for production environments.

### pipenv

In this session we're going to make virtual environment for our project. So Let's start this session to get to know what is a virtual environment and how to make it.
- Every time we're running a file from a directory we're using the executive files from a global directory. For example when we install python on our machine the executable files that are able to run our codes will go to somewhere like _/home/username/python/bin/_ for example the pip command may go to _/home/username/python/bin/pip_.
- Sometimes the versions of libraries conflict (the project may not run or get into massive errors). For example we have an old project that uses sklearn library with the version of 0.24.1 and now we want to run it using sklearn version 1.0.0. We may get into errors because of the version conflict.
   - To solve the conflict we can make virtual environments. Virtual environment is something that can seperate the libraries installed in our system and the libraries with specified version we want our project to run with. There are a lot of ways to create a virtual environments. One way we are going to use is using a library named pipenv.
   - pipenv is a library that can create a virutal environment. To install this library just use the classic method ```pip install pipenv```.
   - After installing pipenv we must to install the libraries we want for our project in the new virtual environment. It's really easy, Just use the command pipenv instead of pip. ```pipenv install numpy sklearn==0.24.1 flask```. With this command we installed the libraries we want for our project.
   - Note that using the pipenv command we made two files named _Pipfile_ and _Pipfile.lock_. If we look at this files closely we can see that in Pipfile the libraries we installed are named. If we specified the library name, it's also specified in Pipfile.
   - In _Pipfile.lock_ we can see that each library with it's installed version is named and a hash file is there to reproduce if we move the environment to another machine.
   - If we want to run the project in another machine, we can easily installed the libraries we want with the command ```pipenv install```. This command will look into _Pipfile_ and _Pipfile.lock_ to install the libraries with specified version.
   - After installing the required libraries we can run the project in the virtual environment with ```pipenv shell``` command. This will go to the virtual environment's shell and then any command we execute will use the virtual environment's libraries.
- Installing and using the libraries such as gunicorn is the same as the last session.
- Until here we made a virtual environment for our libraries with a required specified version. To seperate this environment more, such as making gunicorn be able to run in windows machines we need another way. The other way is using Docker. Docker allows us to seperate everything more than before and make any project able to run on any machine that support Docker smoothly.
- In the next session we'll go in detail of how Docker works and how to use it.

### Docker

#### Installing Docker
To isolate more our project file from our system machine, there is an option named Docker. With Docker you are able to pack all your project is a system that you want and run it in any system machine. For example if you want Ubuntu 20.4 you can have it in a mac or windows machine or other operating systems. <br>
To get started with Docker for the churn prediction project you can follow the instructions below.

##### Ubuntu 

```bash
sudo apt-get install docker.io
```

To run docker without `sudo`, follow [this instruction](https://docs.docker.com/engine/install/linux-postinstall/).

##### Windows

To install the Docker you can just follow the instruction by Andrew Lock in this link: https://andrewlock.net/installing-docker-desktop-for-windows/

##### MacOS

Follow the steps in the [Docker docs](https://docs.docker.com/desktop/install/mac-install/).


#### Notes

- Once our project was packed in a Docker container, we're able to run our project on any machine.
- First we have to make a Docker image. In Docker image file there are settings and dependecies we have in our project. To find Docker images that you need you can simply search the [Docker](https://hub.docker.com/search?type=image) website.

Here a Dockerfile (There should be no comments in Dockerfile, so remove the comments when you copy)

```docker
# First install the python 3.8, the slim version uses less space
FROM python:3.8.12-slim

# Install pipenv library in Docker 
RUN pip install pipenv

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["*.py", "churn-model.bin", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "churn_serving:app"]
```

The flags `--deploy` and `--system` makes sure that we install the dependencies directly inside the Docker container without creating an additional virtual environment (which `pipenv` does by default). 

If we don't put the last line `ENTRYPOINT`, we will be in a python shell.
Note that for the entrypoint, we put our commands in double quotes.

After creating the Dockerfile, we need to build it:

```bash
docker build -t churn-prediction .
```

To run it,  execute the command below:

```bash
docker run -it -p 9696:9696 churn-prediction:latest
```

Flag explanations: 

- `-t`: is used for specifying the tag name "churn-prediction".
- `-it`: in order for Docker to allow us access to the terminal.
- `--rm`: allows us to remove the image from the system after we're done.  
- `-p`: to map the 9696 port of the Docker to 9696 port of our machine. (first 9696 is the port number of our machine and the last one is Docker container port.)
- `--entrypoint=bash`: After running Docker, we will now be able to communicate with the container using bash (as you would normally do with the Terminal). Default is `python`.


At last you've deployed your prediction app inside a Docker continer. Congratulations 🥳


### AWS Elastic Beanstalk

<a href="https://www.youtube.com/watch?v=HGPJ4ekhcLg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR"><img src="images/thumbnail-5-07.jpg"></a>

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)

### Heroku

Here we will learn how to deploy our apps in heroku instead of AWS.
- First of all create your web service with flask. (example file: [churn_prediction.py](https://github.com/amindadgar/customer-churn-app/blob/main/churn_serving.py)
- Then create a file named _requirements.txt_ and pass your dependencies there. Example:
 ```
 pickle
 numpy
 flask
 gunicorn
  ```
- Create another file named _Procfile_ and add the app you want to be able to run there. Example:
 ```
web: gunicorn churn_serving:app
  ```
  Note that the churn_serving name in the box above is the name of the main python file we're going to be running.
 - Create your heroku profile, Go to dashboard and the Deploy tab.
 - Follow the instruction to Deploy using Heroku Git.
 - Great, your app is now available from global universe.

I've put my heroku app files in this repository:
https://github.com/amindadgar/customer-churn-app 


### Summary

In this chapter we learned this topics:
- We learned how to save the model and load it to re-use it without running the previous code.
- How to deploy the model in a web service.
- How to create a virtual environment.
- How to create a container and run our code in any operating systems.
- How to implement our code in a public web service and aceess it from outside a local computer.

In the next chapter we would learn the algorithms such as Decision trees, Random forests and Gradient boosting as an alternative way of combining decision tress.

### Using conda

### Using Conda

- Create a new environment and install the packages:
```bash
conda create -n churn-model python=3.8
conda activate churn-model

conda install scikit-learn flask pandas gunicorn

# Create environment.yml file
# If there are issues keep only the packages you need in the file
conda env export > environment.yml
```

- For instance for the `churn-model` environment, the `environment.yml` file looks like this:

> There were issues resolving the dependencies as exported by `conda env export`. I had to keep only the packages I needed.

```yaml
name: churn-model
channels:
  - defaults
dependencies:
  - flask
  - python=3.8
  - scikit-learn=1.3.0
  - pip:
      - gunicorn
```

- The Dockerfile looks like this:

```docker
FROM continuumio/miniconda3

WORKDIR /app

# Create a Conda environment using environment.yml
COPY environment.yml /app/
RUN conda env create -f environment.yml

# Copy the model files
COPY ["model_C=1.0.bin", "predict.py", "./"]

# The code to run when container is started:
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh
EXPOSE 9696
ENTRYPOINT ["./entrypoint.sh"]
```
- And entrypoint.sh looks like this:

```bash
#!/bin/bash --login
# The --login ensures the bash configuration is loaded,

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate churn-model
# enable strict mode:
set -euo pipefail

# exec the final command:
gunicorn --bind=0.0.0.0:9696 predict:app
```

- Docker commands
```bash
docker build -t churn-model .
docker run -it -p 9696:9696 churn-model:latest
```
- Test the web service `bash python predict-test.py`
- Install awsebcli: `pip install awsebcli`
- Create a new Elastic Beanstalk application and follow instructions: `eb init -i`
- Test locally: `eb local run --port 9696`
- Due to memory issues with miniconda3, you should use a larger instance type: `eb create churn-serving-env --instance-type t2.medium`
> Make sure to use custom settings to limit the access to the web service in a real application.