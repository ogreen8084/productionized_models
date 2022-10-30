from flask import Flask, request
import pickle
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

model_pk = pickle.load(open("heat_load.pkl", "rb"))


@app.route("/api_predict", methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        return "Please send Post Request"

    elif request.method == "POST":
        data = request.get_json()
        compactness = data["COMPACTNESS"]
        surface_area = data["SURFACE_AREA"]
        wall_area = data["WALL_AREA"]
        roof_area = data["ROOF_AREA"]
        height = data["HEIGHT"]
        orientation = data["ORIENTATION"]
        glazing_area = data["GLAZING_AREA"]
        glazing_area_dist = data["GLAZING_AREA_DIST"]

        input1 = pd.DataFrame(
            [
                compactness,
                surface_area,
                wall_area,
                roof_area,
                height,
                orientation,
                glazing_area,
                glazing_area_dist,
            ]
        )

        input1 = input1.transpose()
        input1.columns = [
            "COMPACTNESS",
            "SURFACE_AREA",
            "WALL_AREA",
            "ROOF_AREA",
            "HEIGHT",
            "ORIENTATION",
            "GLAZING_AREA",
            "GLAZING_AREA_DIST",
        ]

        prediction = model_pk.predict(input1)

        return str(prediction)


app.run()
