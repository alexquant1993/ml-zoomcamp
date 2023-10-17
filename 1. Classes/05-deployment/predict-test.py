import requests
host = 'churn-serving-env2.eba-qidvybpx.us-west-2.elasticbeanstalk.com'
url = f'http://{host}/predict'
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 12,
    'monthlycharges': 29.85,
    'totalcharges': (12 * 29.85)
}

response = requests.post(url, json=customer).json()

if response['churn'] == True:
    print('sending promo email to %s' % 'xyz-123')
else:
    print('not sending promo email to %s' % 'xyz-123')
