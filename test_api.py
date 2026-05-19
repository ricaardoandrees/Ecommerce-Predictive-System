import requests

data = {
    "Administrative": 3,
    "Administrative_Duration": 61.0000,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 15,
    "ProductRelated_Duration": 504.506000,
    "BounceRates": 0.00000,
    "ExitRates": 0.026984,
    "PageValues": 11.535959,
    "SpecialDay": 0.0,
    "Month": "Apr",
    "OperatingSystems": 1,
    "Browser": 2,
    "Region": 4,
    "TrafficType": 4,
    "VisitorType": "Returning_Visitor",
    "Weekend": 1
}

try:
    r = requests.post('http://localhost:5000/predict', json=data)
    print(r.status_code)
    print(r.json())
except Exception as e:
    print(f"Error: {e}")
