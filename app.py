from flask import Flask, render_template, request, redirect
from pymongo import MongoClient
import joblib
import pandas as pd
from datetime import datetime


app = Flask(__name__)


# =========================================================
# Load trained ML model
# =========================================================

model = joblib.load("ml/disaster_model.pkl")

# Load preprocessing pipeline
preprocessor = joblib.load("ml/preprocessor.pkl")


# =========================================================
# MongoDB Connection
# =========================================================

client = MongoClient("mongodb://localhost:27017")

db = client["ai_disaster_db"]

disasters = db["disasters"]


# =========================================================
# Home / Disaster Report Page
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# Report Disaster
# =========================================================

@app.route("/report-disaster", methods=["POST"])
def report_disaster():

    # -----------------------------------------------------
    # Collect form data
    # -----------------------------------------------------

    disaster_data = pd.DataFrame([{

        "disaster_type": request.form["name"],

        "description": request.form["description"],

        "location": request.form["location"],

        "people_affected": int(
            request.form["people_affected"]
        ),

        "duration_hours": int(
            request.form["duration_hours"]
        ),

        "infrastructure_damage":
            request.form["infrastructure_damage"],

        "weather_condition":
            request.form["weather_condition"],

        "evacuation_required":
            request.form["evacuation_required"]

    }])


    # -----------------------------------------------------
    # Preprocess input
    # -----------------------------------------------------

    processed_data = preprocessor.transform(
        disaster_data
    )


    # -----------------------------------------------------
    # Predict severity using ML model
    # -----------------------------------------------------

    predicted_severity = model.predict(
        processed_data
    )[0]


    # -----------------------------------------------------
    # Determine priority
    # -----------------------------------------------------

    priority = (

        "High"

        if predicted_severity in ["High", "Critical"]

        else "Normal"

    )


    # -----------------------------------------------------
    # Base response recommendations
    # -----------------------------------------------------

    response_recommendations = {

        "Critical":
            "Immediate emergency response required. "
            "Evacuate affected people and deploy rescue teams.",

        "High":
            "Urgent response required. "
            "Send emergency teams and prepare evacuation support.",

        "Medium":
            "Monitor the situation closely and "
            "prepare local emergency resources.",

        "Low":
            "Continue monitoring and provide basic "
            "assistance if required."

    }


    response = response_recommendations.get(

        predicted_severity,

        "Monitor the situation and assess emergency needs."

    )


    # -----------------------------------------------------
    # Additional response recommendations
    # -----------------------------------------------------

    people_affected = int(
        request.form["people_affected"]
    )

    evacuation_required = request.form[
        "evacuation_required"
    ]

    infrastructure_damage = request.form[
        "infrastructure_damage"
    ]


    # Evacuation recommendation

    if evacuation_required == "Yes":

        response += (
            " Evacuation support should be prepared immediately."
        )


    # Large affected population

    if people_affected >= 1000:

        response += (
            " Large numbers of affected people require "
            "additional shelter and relief resources."
        )


    # Severe infrastructure damage

    if infrastructure_damage == "High":

        response += (
            " Infrastructure damage is severe; "
            "coordinate repair and emergency access teams."
        )


    # -----------------------------------------------------
    # Create disaster record
    # -----------------------------------------------------

    disaster = {

        "name": request.form["name"],

        "location": request.form["location"],

        "severity": predicted_severity,

        "description": request.form["description"],

        "status": "active",

        "priority": priority,

        "response": response,

        "people_affected": people_affected,

        "duration_hours": int(
            request.form["duration_hours"]
        ),

        "infrastructure_damage":
            infrastructure_damage,

        "weather_condition":
            request.form["weather_condition"],

        "evacuation_required":
            evacuation_required

    }


    # -----------------------------------------------------
    # Save disaster to MongoDB
    # -----------------------------------------------------

    disasters.insert_one(disaster)


    return redirect("/dashboard")


# =========================================================
# Dashboard
# =========================================================

@app.route("/dashboard")
def dashboard():

    # -----------------------------------------------------
    # Get filters
    # -----------------------------------------------------

    status_filter = request.args.get("status")

    priority_filter = request.args.get("priority")


    # -----------------------------------------------------
    # Build MongoDB query
    # -----------------------------------------------------

    query = {}


    if status_filter in ["active", "resolved"]:

        query["status"] = status_filter


    if priority_filter in ["High", "Normal"]:

        query["priority"] = priority_filter


    # -----------------------------------------------------
    # Get disaster records
    # -----------------------------------------------------

    disaster_list = list(
        disasters.find(query).sort("_id", -1)
    )


        # -----------------------------------------------------
    # General statistics
    # -----------------------------------------------------

    total_disasters = disasters.count_documents({})

    active_disasters = disasters.count_documents({
        "status": "active"
    })

    critical_disasters = disasters.count_documents({
        "severity": "Critical"
    })

    high_disasters = disasters.count_documents({
        "severity": "High"
    })

    medium_disasters = disasters.count_documents({
        "severity": "Medium"
    })

    low_disasters = disasters.count_documents({
        "severity": "Low"
    })

    # -----------------------------------------------------
    # AI Risk Overview statistics
    # -----------------------------------------------------

    low_disasters = disasters.count_documents({
        "severity": "Low"
    })

        # -----------------------------------------------------
    # Disaster type statistics
    # -----------------------------------------------------

    disaster_types = disasters.aggregate([
        {
            "$group": {
                "_id": "$name",
                "count": {"$sum": 1}
            }
        },
        {
            "$sort": {
                "count": -1
            }
        }
    ])

    disaster_type_data = list(disaster_types)

    medium_disasters = disasters.count_documents({
        "severity": "Medium"
    })

    high_disasters = disasters.count_documents({
        "severity": "High"
    })


    # -----------------------------------------------------
    # Send data to dashboard.html
    # -----------------------------------------------------

    return render_template(
    "dashboard.html",

    disasters=disaster_list,

    total_disasters=total_disasters,

    active_disasters=active_disasters,

    critical_disasters=critical_disasters,

    high_disasters=high_disasters,

    medium_disasters=medium_disasters,

    low_disasters=low_disasters,

    disaster_type_data=disaster_type_data
)


# =========================================================
# Resolve Disaster
# =========================================================

@app.route(
    "/resolve-disaster/<disaster_id>",
    methods=["POST"]
)
def resolve_disaster(disaster_id):

    from bson.objectid import ObjectId


    disasters.update_one(

        {
            "_id": ObjectId(disaster_id)
        },

        {
            "$set": {
                "status": "resolved"
            }
        }

    )


    return redirect("/dashboard")


# =========================================================
# Run Flask Application
# =========================================================

if __name__ == "__main__":

    app.run(debug=False)