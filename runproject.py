from flask import Flask, render_template, url_for, request, send_from_directory
import numpy as np
import pandas as pd
import folium
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/stats")
def stats():
    return render_template("stats.html")

@app.route('/storage/<path:x>')
def storage(x):
    return send_from_directory("storage", x)

@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        input = request.form
        #Region
        region = input["geo"]
        strRegion = ""
        if region == "fra":
            fra = 1
            ger = 0
            spa = 0
            strRegion = "France"
        if region == "ger":
            fra = 0
            ger = 1
            spa = 0
            strRegion = "German"
        if region == "spa":
            fra = 0
            ger = 0
            spa = 1
            strRegion = "Spain"
        #Gender
        gender = input["gender"]
        strGender = ""
        if gender == "m":
            mal = 1
            fem = 0
            strGender = "Male"
        if gender == "f":
            mal = 0
            fem = 1
            strGender = "Female"
        #CCard
        ccard = input["ccard"]
        strCard = ""
        if ccard == "y":
            cc = 1
            strCard = "Yes"
        else:
            cc = 0
            strCard = "No"
        #Active
        active = input["active"]
        strAct = ""
        if active == "y":
            act = 1
            strAct = "Yes"
        else:
            act = 0
            strAct = "No"
        #Product
        prod = int(input["prod"])
        #Age
        age = int(input["age"])
        #Tenure
        ten = int(input["tenure"])
        #Credit
        crd = int(input["credit"])
        #Salary
        sal = float(input["salary"])
        #Balance
        bal = float(input["balance"])
        #Result
        datainput = [[fra, ger, spa, fem, mal, crd, age, ten, bal, prod, cc, act, sal]]
        pred = model.predict(datainput)[0]
        proba = model.predict_proba(datainput)[0]
        if pred == 0:
            prbb = round((proba[0]*100), 1)
            rslt = "RETAIN"
            color = "green"
        else:
            prbb = round((proba[1]*100), 1)
            rslt = "EXIT"
            color = "red"
        return render_template(
            "result.html", region= strRegion, gender= strGender,
            credit= crd, age= age, tenure= ten, balance= bal,
            product= prod, ccard = strCard, active= strAct, salary= sal,
            result= rslt, proba = prbb, color = color
        )


if __name__ == "__main__":
    model = joblib.load("modelFix")
    app.run(debug=True, port=5050)