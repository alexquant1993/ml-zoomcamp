import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
## AWS API GATEWAY - ALREADY TESTED AND REMOVED
# url = "https://r151tir4q9.execute-api.eu-south-2.amazonaws.com/test/predict"

data = {"url": "http://bit.ly/mlbookcamp-pants"}

result = requests.post(url, json=data).json()
print(result)
