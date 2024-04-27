# initializing and training the model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import threading


# load model
model = keras.models.load_model("model.keras")


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


def get_csv_data():
    data = pd.read_csv("CVD_cleaned.csv")
    Y = data["Heart_Disease"]
    X = data.drop(
        [
            "General_Health",
            "Skin_Cancer",
            "Other_Cancer",
            "Depression",
            "Diabetes",
            "Arthritis",
            "Heart_Disease",
        ],
        axis=1,
    )
    # one hot encoding for checkup column adding 5 columns checkup_past_year, checkup_past_2years, checkup_past_5years, checkup_more_5years, checkup_never
    checkup = pd.get_dummies(X["Checkup"])

    # reorder columns to have checkup_never at the end
    checkup = checkup[
        [
            "Within the past year",
            "Within the past 2 years",
            "Within the past 5 years",
            "5 or more years ago",
            "Never",
        ]
    ]

    # rename columns to have checkup_ prefix
    checkup.columns = [
        "checkup_past_year",
        "checkup_past_2years",
        "checkup_past_5years",
        "checkup_more_5years",
        "checkup_never",
    ]

    # drop the original checkup column
    X = X.drop("Checkup", axis=1)

    # make all false values 0 and all true values 1
    checkup = checkup * 1

    # concatenate the one hot encoded checkup columns
    X = pd.concat([X, checkup], axis=1)

    Age_Category = pd.get_dummies(X["Age_Category"])
    # rename columns to have Age_ prefix
    Age_Category.columns = [
        "Age_18-24",
        "Age_25-29",
        "Age_30-34",
        "Age_35-39",
        "Age_40-44",
        "Age_45-49",
        "Age_50-54",
        "Age_55-59",
        "Age_60-64",
        "Age_65-69",
        "Age_70-74",
        "Age_75-79",
        "Age_80+",
    ]

    # drop the original Age_Category column
    X = X.drop("Age_Category", axis=1)

    # make all false values 0 and all true values 1
    Age_Category = Age_Category * 1

    # concatenate the one hot encoded Age_Category columns
    X = pd.concat([X, Age_Category], axis=1)

    X["Exercise"] = X["Exercise"].map({"Yes": 1, "No": 0})
    X["Smoking_History"] = X["Smoking_History"].map({"Yes": 1, "No": 0})

    X["Sex"] = X["Sex"].map({"Female": 1, "Male": 0})

    Y = Y.map({"Yes": 1, "No": 0})

    return X, Y


def train_model():
    global model
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
    newCheckup = []
    for i in range(len(checkup)):
        if checkup[i] == 1:
            newCheckup.append([1, 0, 0, 0, 0])
        elif checkup[i] == 2:
            newCheckup.append([0, 1, 0, 0, 0])
        elif checkup[i] == 3:
            newCheckup.append([0, 0, 1, 0, 0])
        elif checkup[i] == 4:
            newCheckup.append([0, 0, 0, 1, 0])
        elif checkup[i] == 5:
            newCheckup.append([0, 0, 0, 0, 1])

    newAge = []
    for i in range(len(age)):
        if age[i] == 1:
            newAge.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif age[i] == 2:
            newAge.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif age[i] == 3:
            newAge.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif age[i] == 4:
            newAge.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif age[i] == 5:
            newAge.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif age[i] == 6:
            newAge.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif age[i] == 7:
            newAge.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif age[i] == 8:
            newAge.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif age[i] == 9:
            newAge.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif age[i] == 10:
            newAge.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif age[i] == 11:
            newAge.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif age[i] == 12:
            newAge.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif age[i] == 13:
            newAge.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    X = pd.concat(
        [X, pd.DataFrame(newCheckup), pd.DataFrame(newAge)],
        axis=1,
        ignore_index=True,
    )

    X.columns = [
        "Exercise",
        "Sex",
        "Height_(cm)",
        "Weight_(kg)",
        "BMI",
        "Smoking_History",
        "Alcohol_Consumption",
        "Fruit_Consumption",
        "Green_Vegetables_Consumption",
        "FriedPotato_Consumption",
        "checkup_past_year",
        "checkup_past_2years",
        "checkup_past_5years",
        "checkup_more_5years",
        "checkup_never",
        "Age_18-24",
        "Age_25-29",
        "Age_30-34",
        "Age_35-39",
        "Age_40-44",
        "Age_45-49",
        "Age_50-54",
        "Age_55-59",
        "Age_60-64",
        "Age_65-69",
        "Age_70-74",
        "Age_75-79",
        "Age_80+",
    ]

    Y = data["truth"]

    X_data, Y_data = get_csv_data()
    # add the new data to the old data vertically
    X = pd.concat([X, X_data], ignore_index=True)
    Y = pd.concat([Y, Y_data], ignore_index=True)

    X["Height_(cm)"] = (X["Height_(cm)"] - X["Height_(cm)"].mean()) / X[
        "Height_(cm)"
    ].std()
    X["Weight_(kg)"] = (X["Weight_(kg)"] - X["Weight_(kg)"].mean()) / X[
        "Weight_(kg)"
    ].std()
    X["BMI"] = (X["BMI"] - X["BMI"].mean()) / X["BMI"].std()

    # the Fruit_Consumption, Green_Vegtables_Consumption, FriedPotato_Consumption, Alcohol_Consumption colums are exponentially distributed
    X["Fruit_Consumption"] = np.log(X["Fruit_Consumption"] + 1)
    X["Green_Vegetables_Consumption"] = np.log(X["Green_Vegetables_Consumption"] + 1)
    X["FriedPotato_Consumption"] = np.log(X["FriedPotato_Consumption"] + 1)
    X["Alcohol_Consumption"] = np.log(X["Alcohol_Consumption"] + 1)

    X["Fruit_Consumption"] = (
        X["Fruit_Consumption"] - X["Fruit_Consumption"].mean()
    ) / X["Fruit_Consumption"].std()
    X["Green_Vegetables_Consumption"] = (
        X["Green_Vegetables_Consumption"] - X["Green_Vegetables_Consumption"].mean()
    ) / X["Green_Vegetables_Consumption"].std()
    X["FriedPotato_Consumption"] = (
        X["FriedPotato_Consumption"] - X["FriedPotato_Consumption"].mean()
    ) / X["FriedPotato_Consumption"].std()
    X["Alcohol_Consumption"] = (
        X["Alcohol_Consumption"] - X["Alcohol_Consumption"].mean()
    ) / X["Alcohol_Consumption"].std()

    y_model = keras.utils.to_categorical(Y, 2)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(7, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    model.compile(
        metrics=[keras.metrics.F1Score()],
        loss="categorical_crossentropy",
        optimizer="rmsprop",
    )

    # give more weight to the positive class
    class_weight = {0: 1, 1: 7}

    model.fit(
        X,
        y_model,
        epochs=15,
        verbose=0,
        class_weight=class_weight,
        batch_size=32,
    )

    model.save("model.keras")


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
    global model
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
        return {"prediction": model_prediction[0]}


@app.route("/truth", methods=["GET", "POST"])
def truth():
    global model
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
        # if there are 10 new rows train the model
        mycursor = mydb.cursor()
        mycursor.execute("SELECT COUNT(*) FROM health_data;")
        myresult = mycursor.fetchall()
        if myresult[0][0] % 10 == 0:
            threading.Thread(target=train_model).start()

        return {"status": "success"}


@app.route("/get_data", methods=["GET"])
def get_data():
    mycursor = mydb.cursor(dictionary=True)
    mycursor.execute("SELECT * FROM health_data;")
    myresult = mycursor.fetchall()
    return myresult


# main driver function
if __name__ == "__main__":

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
