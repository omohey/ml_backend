# initializing and training the model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# print("Here")

# load model
model = keras.models.load_model("model.keras")

print("Model Trained")


# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, redirect, url_for, request

# import module for sql database
import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()


# create a connection to the database
mydb = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_DATABASE"),
)


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# make cors work
from flask_cors import CORS

# allow all origins and all headers and methods
CORS(app)


def one_hot_encode(age, checkup):
    # age and checkup should both be one hot encoded before being passed to the model
    # checkup have values from 1 to 5
    if checkup == 1:
        checkup = [1, 0, 0, 0, 0]
    elif checkup == 2:
        checkup = [0, 1, 0, 0, 0]
    elif checkup == 3:
        checkup = [0, 0, 1, 0, 0]
    elif checkup == 4:
        checkup = [0, 0, 0, 1, 0]
    elif checkup == 5:
        checkup = [0, 0, 0, 0, 1]

    # age have values from 1 to 13
    if age == 1:
        age = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif age == 2:
        age = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif age == 3:
        age = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif age == 4:
        age = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif age == 5:
        age = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif age == 6:
        age = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif age == 7:
        age = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif age == 8:
        age = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif age == 9:
        age = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif age == 10:
        age = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif age == 11:
        age = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif age == 12:
        age = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif age == 13:
        age = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    return age, checkup


def normalize_data(data):
    # normalize the data
    height = data["height"]
    weight = data["weight"]
    bmi = data["bmi"]
    exercise = data["exercise"]
    gender = data["gender"]
    smoker = data["smoker"]
    fruit = data["fruit"]
    greenVeg = data["greenVeg"]
    friedPotato = data["friedPotato"]
    alcohol = data["alcohol"]

    # convert to float
    height = float(height)
    weight = float(weight)
    bmi = float(bmi)

    # normalize the data
    height = (height - 170.61524862880196) / 10.658026052142686
    weight = (weight - 83.58865454227563) / 21.343209763039585
    bmi = (bmi - 28.62621053960772) / 6.522323094694673

    # convert to int
    exercise = int(exercise)
    gender = int(gender)
    smoker = int(smoker)
    fruit = int(fruit)
    greenVeg = int(greenVeg)
    friedPotato = int(friedPotato)
    alcohol = int(alcohol)

    # log the rest first
    fruit = np.log(fruit + 1)
    greenVeg = np.log(greenVeg + 1)
    friedPotato = np.log(friedPotato + 1)
    alcohol = np.log(alcohol + 1)

    fruit = (fruit - 3.0386082652038517) / 1.0052688888508805
    greenVeg = (greenVeg - 2.355472713643299) / 1.0173698710078443
    friedPotato = (friedPotato - 1.5532744435852295) / 0.9339378339487504
    alcohol = (alcohol - 1.0657940216855384) / 1.1711258843198782

    data = {
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "exercise": exercise,
        "gender": gender,
        "smoker": smoker,
        "fruit": fruit,
        "greenVeg": greenVeg,
        "friedPotato": friedPotato,
        "alcohol": alcohol,
    }

    return data


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route("/")
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return "Hello World"


@app.route("/predict", methods=["GET", "POST"])
# if method is post add the data to the database
def form():
    print("Form")
    if request.method == "POST":
        print("Post")
        # get the data from the body of the request
        height = request.json["height"]
        weight = request.json["weight"]
        bmi = request.json["bmi"]
        alcohol = request.json["alcohol"]
        fruit = request.json["fruit"]
        greenVeg = request.json["greenVeg"]
        friedPotato = request.json["friedPotato"]
        age = request.json["age"]
        checkup = request.json["checkup"]
        exercise = request.json["exercise"]
        gender = request.json["gender"]
        smoker = request.json["smoker"]

        print(
            "Data: ",
            height,
            weight,
            bmi,
            alcohol,
            fruit,
            greenVeg,
            friedPotato,
            age,
            checkup,
            exercise,
            gender,
            smoker,
        )

        # one hot encode the age and checkup
        age, checkup = one_hot_encode(age, checkup)

        # normalize the data
        data = {
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "exercise": exercise,
            "gender": gender,
            "smoker": smoker,
            "fruit": fruit,
            "greenVeg": greenVeg,
            "friedPotato": friedPotato,
            "alcohol": alcohol,
        }

        data = normalize_data(data)

        height = data["height"]
        weight = data["weight"]
        bmi = data["bmi"]
        exercise = data["exercise"]
        gender = data["gender"]
        smoker = data["smoker"]
        fruit = data["fruit"]
        greenVeg = data["greenVeg"]
        friedPotato = data["friedPotato"]
        alcohol = data["alcohol"]

        # make the prediction
        data = [
            exercise,
            gender,
            height,
            weight,
            bmi,
            smoker,
            alcohol,
            fruit,
            greenVeg,
            friedPotato,
        ]
        data = data + checkup + age
        data = np.array(data).reshape(1, -1)

        model_prediction = model.predict(data, verbose=0)
        model_prediction = [1 if x[1] > x[0] else 0 for x in model_prediction]
        print(model_prediction)
        return {"prediction": model_prediction[0]}


@app.route("/truth", methods=["GET", "POST"])
def truth():
    if request.method == "POST":
        # get the data from the body of the request
        height = request.json["height"]
        weight = request.json["weight"]
        bmi = request.json["bmi"]
        alcohol = request.json["alcohol"]
        fruit = request.json["fruit"]
        greenVeg = request.json["greenVeg"]
        friedPotato = request.json["friedPotato"]
        age = request.json["age"]
        checkup = request.json["checkup"]
        exercise = request.json["exercise"]
        gender = request.json["gender"]
        smoker = request.json["smoker"]
        truth = request.json["truth"]

        # # add the data to the database
        mycursor = mydb.cursor()
        sql = """INSERT INTO health_data (height, weight, BMI, alcohol_Consumption, fruit_Consumption, green_Vegtable_Consumption, fried_Potato_Consumption, age, checkup, isExercise, isFemale, isSmoker, truth)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        val = (
            height,
            weight,
            bmi,
            alcohol,
            fruit,
            greenVeg,
            friedPotato,
            age,
            checkup,
            exercise,
            gender,
            smoker,
            truth,
        )
        mycursor.execute(sql, val)
        mydb.commit()
        # print id of inserted record
        print(mycursor.lastrowid)
        # if row id is a multiple of 5 use online learning to update the model
        if mycursor.lastrowid % 5 == 0:
            # get the data from the database
            mycursor = mydb.cursor()
            mycursor.execute("SELECT * FROM health_data;")
            myresult = mycursor.fetchall()
            data = pd.DataFrame(myresult)
            data.columns = [
                "id",
                "height",
                "weight",
                "bmi",
                "alcohol",
                "fruit",
                "greenVeg",
                "friedPotato",
                "age",
                "checkup",
                "exercise",
                "female",
                "smoker",
                "truth",
            ]
            X = data.copy()
            X = X.drop(["truth"], axis=1)
            # correct ordering of columns is exercise, female, height, weight, bmi, smoker, alcohol, fruit, greenVeg, friedPotato, checkup, age
            X = X[
                [
                    "exercise",
                    "female",
                    "height",
                    "weight",
                    "bmi",
                    "smoker",
                    "alcohol",
                    "fruit",
                    "greenVeg",
                    "friedPotato",
                    "checkup",
                    "age",
                ]
            ]
            # checkup and age should be one hot encoded
            checkup = X["checkup"]
            age = X["age"]
            X = X.drop(["checkup", "age"], axis=1)
            for i in range(len(checkup)):
                checkup[i] = (
                    [1, 0, 0, 0, 0]
                    if checkup[i] == 1
                    else (
                        [0, 1, 0, 0, 0]
                        if checkup[i] == 2
                        else (
                            [0, 0, 1, 0, 0]
                            if checkup[i] == 3
                            else [0, 0, 0, 1, 0] if checkup[i] == 4 else [0, 0, 0, 0, 1]
                        )
                    )
                )
            for i in range(len(age)):
                age[i] = (
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    if age[i] == 1
                    else (
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        if age[i] == 2
                        else (
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            if age[i] == 3
                            else (
                                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                if age[i] == 4
                                else (
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                                    if age[i] == 5
                                    else (
                                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                                        if age[i] == 6
                                        else (
                                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                                            if age[i] == 7
                                            else (
                                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                                                if age[i] == 8
                                                else (
                                                    [
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        1,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                    ]
                                                    if age[i] == 9
                                                    else (
                                                        [
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            1,
                                                            0,
                                                            0,
                                                            0,
                                                        ]
                                                        if age[i] == 10
                                                        else (
                                                            [
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                0,
                                                                1,
                                                                0,
                                                                0,
                                                            ]
                                                            if age[i] == 11
                                                            else (
                                                                [
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    1,
                                                                    0,
                                                                ]
                                                                if age[i] == 12
                                                                else [
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    1,
                                                                ]
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            X = pd.concat(
                [X, pd.DataFrame(checkup), pd.DataFrame(age)], axis=1, ignore_index=True
            )
            Y = data["truth"]
            y_model = keras.utils.to_categorical(Y, 2)

            # need to normalize

        return {"status": "success"}


@app.route("/get_data", methods=["GET"])
def get_data():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM health_data ORDER BY ID DESC LIMIT 5;")
    myresult = mycursor.fetchall()
    return {"data": myresult}


# main driver function
if __name__ == "__main__":

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
