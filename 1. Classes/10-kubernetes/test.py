import requests

# url = "http://localhost:9696/predict"
# url = "http://localhost:8080/predict"
url = "http://a2f8aac77a0cd47f08b0530ddbd929c4-737227248.eu-south-2.elb.amazonaws.com/predict"

data = {"url": "http://bit.ly/mlbookcamp-pants"}

result = requests.post(url, json=data).json()
print(result)
