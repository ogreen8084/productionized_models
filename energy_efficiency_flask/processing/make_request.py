import requests

url = "http://127.0.0.1:5000/api_predict"
data = {
    "COMPACTNESS": 0.86,
    "SURFACE_AREA": 588,
    "WALL_AREA": 294,
    "ROOF_AREA": 147,
    "HEIGHT": 7,
    "ORIENTATION": 4,
    "GLAZING_AREA": 0.25,
    "GLAZING_AREA_DIST": 4,
}


r = requests.post(url, json=data)
print("The prediction is: ", r.text)
