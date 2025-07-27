# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle
import os
import numpy as np

# Initialize Flask app with custom template folder
app = Flask(__name__, template_folder='.')

# Load the CSV file with error handling
try:
    df_1 = pd.read_csv("first_telc.csv")
    print("CSV file loaded successfully")
except FileNotFoundError:
    print("Warning: first_telc.csv not found. Using empty DataFrame.")
    df_1 = pd.DataFrame()
except Exception as e:
    print(f"Error loading CSV: {e}")
    df_1 = pd.DataFrame()

@app.route("/")
def loadPage():
    return render_template('home.html', 
                         query1="", query2="", query3="", query4="", query5="",
                         query6="", query7="", query8="", query9="", query10="",
                         query11="", query12="", query13="", query14="", query15="",
                         query16="", query17="", query18="", query19="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # Get all form inputs
        inputQuery1 = request.form.get('query1', '0')  # SeniorCitizen
        inputQuery2 = request.form.get('query2', '0')  # MonthlyCharges
        inputQuery3 = request.form.get('query3', '0')  # TotalCharges
        inputQuery4 = request.form.get('query4', 'Male')  # gender
        inputQuery5 = request.form.get('query5', 'No')   # Partner
        inputQuery6 = request.form.get('query6', 'No')   # Dependents
        inputQuery7 = request.form.get('query7', 'No')   # PhoneService
        inputQuery8 = request.form.get('query8', 'No')   # MultipleLines
        inputQuery9 = request.form.get('query9', 'No')   # InternetService
        inputQuery10 = request.form.get('query10', 'No') # OnlineSecurity
        inputQuery11 = request.form.get('query11', 'No') # OnlineBackup
        inputQuery12 = request.form.get('query12', 'No') # DeviceProtection
        inputQuery13 = request.form.get('query13', 'No') # TechSupport
        inputQuery14 = request.form.get('query14', 'No') # StreamingTV
        inputQuery15 = request.form.get('query15', 'No') # StreamingMovies
        inputQuery16 = request.form.get('query16', 'Month-to-month') # Contract
        inputQuery17 = request.form.get('query17', 'No') # PaperlessBilling
        inputQuery18 = request.form.get('query18', 'Electronic check') # PaymentMethod
        inputQuery19 = request.form.get('query19', '1')  # tenure

        # Check if model file exists
        if not os.path.exists("model.sav"):
            return render_template('home.html', 
                                 output1="Error: Model file (model.sav) not found",
                                 output2="Please ensure the trained model file exists in the project directory",
                                 query1=inputQuery1, query2=inputQuery2, query3=inputQuery3,
                                 query4=inputQuery4, query5=inputQuery5, query6=inputQuery6,
                                 query7=inputQuery7, query8=inputQuery8, query9=inputQuery9,
                                 query10=inputQuery10, query11=inputQuery11, query12=inputQuery12,
                                 query13=inputQuery13, query14=inputQuery14, query15=inputQuery15,
                                 query16=inputQuery16, query17=inputQuery17, query18=inputQuery18,
                                 query19=inputQuery19)

        # Load the trained model
        try:
            model = pickle.load(open("model.sav", "rb"))
        except Exception as e:
            return render_template('home.html', 
                                 output1=f"Error loading model: {str(e)}",
                                 output2="The model file might be corrupted or incompatible",
                                 query1=inputQuery1, query2=inputQuery2, query3=inputQuery3,
                                 query4=inputQuery4, query5=inputQuery5, query6=inputQuery6,
                                 query7=inputQuery7, query8=inputQuery8, query9=inputQuery9,
                                 query10=inputQuery10, query11=inputQuery11, query12=inputQuery12,
                                 query13=inputQuery13, query14=inputQuery14, query15=inputQuery15,
                                 query16=inputQuery16, query17=inputQuery17, query18=inputQuery18,
                                 query19=inputQuery19)

        # Convert and validate numeric inputs
        try:
            # Handle empty strings for numeric fields
            inputQuery1 = float(inputQuery1) if inputQuery1.strip() else 0.0
            inputQuery2 = float(inputQuery2) if inputQuery2.strip() else 0.0
            inputQuery3 = float(inputQuery3) if inputQuery3.strip() else 0.0
            inputQuery19 = int(inputQuery19) if inputQuery19.strip() else 1
        except (ValueError, AttributeError) as e:
            return render_template('home.html', 
                                 output1="Error: Invalid numeric input",
                                 output2="Please ensure SeniorCitizen, MonthlyCharges, TotalCharges, and Tenure are valid numbers",
                                 query1=inputQuery1, query2=inputQuery2, query3=inputQuery3,
                                 query4=inputQuery4, query5=inputQuery5, query6=inputQuery6,
                                 query7=inputQuery7, query8=inputQuery8, query9=inputQuery9,
                                 query10=inputQuery10, query11=inputQuery11, query12=inputQuery12,
                                 query13=inputQuery13, query14=inputQuery14, query15=inputQuery15,
                                 query16=inputQuery16, query17=inputQuery17, query18=inputQuery18,
                                 query19=inputQuery19)

        # Create data array for prediction
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
                 inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
                 inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
        
        # Create DataFrame with the new data
        new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure'])
        
        # Combine with existing data if available
        if not df_1.empty:
            df_2 = pd.concat([df_1, new_df], ignore_index=True)
        else:
            df_2 = new_df.copy()
        
        # Create tenure groups
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        
        try:
            df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        except Exception as e:
            # Use a default tenure group if binning fails
            df_2['tenure_group'] = '1 - 11'
        
        # Drop the original tenure column
        df_2.drop(columns=['tenure'], axis=1, inplace=True)   
        
        # Create dummy variables for categorical features
        categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
        
        new_df_dummies = pd.get_dummies(df_2[categorical_cols])
        
        # Ensure all expected features are present
        for col in model.feature_names_in_:
            if col not in new_df_dummies.columns:
                new_df_dummies[col] = 0

        # Reorder columns to match the model's expected feature order
        new_df_dummies = new_df_dummies[model.feature_names_in_]

        # Make prediction on the last row (new customer data)
        prediction = model.predict(new_df_dummies.tail(1))
        probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

        # Format the results
        if prediction[0] == 1:
            result_text = "This customer is likely to churn (leave)!"
            confidence = f"Confidence: {probability[0] * 100:.2f}%"
        else:
            result_text = "This customer is likely to continue!"
            confidence = f"Confidence: {(1 - probability[0]) * 100:.2f}%"
            
        return render_template('home.html', 
                             output1=result_text, 
                             output2=confidence,
                             query1=inputQuery1, query2=inputQuery2, query3=inputQuery3,
                             query4=inputQuery4, query5=inputQuery5, query6=inputQuery6,
                             query7=inputQuery7, query8=inputQuery8, query9=inputQuery9,
                             query10=inputQuery10, query11=inputQuery11, query12=inputQuery12,
                             query13=inputQuery13, query14=inputQuery14, query15=inputQuery15,
                             query16=inputQuery16, query17=inputQuery17, query18=inputQuery18,
                             query19=inputQuery19)
    
    except Exception as e:
        # Catch any unexpected errors
        return render_template('home.html', 
                             output1=f"Prediction Error: {str(e)}",
                             output2="Please check your inputs and try again. Make sure all required files are present.",
                             query1=request.form.get('query1', ''),
                             query2=request.form.get('query2', ''),
                             query3=request.form.get('query3', ''),
                             query4=request.form.get('query4', ''),
                             query5=request.form.get('query5', ''),
                             query6=request.form.get('query6', ''),
                             query7=request.form.get('query7', ''),
                             query8=request.form.get('query8', ''),
                             query9=request.form.get('query9', ''),
                             query10=request.form.get('query10', ''),
                             query11=request.form.get('query11', ''),
                             query12=request.form.get('query12', ''),
                             query13=request.form.get('query13', ''),
                             query14=request.form.get('query14', ''),
                             query15=request.form.get('query15', ''),
                             query16=request.form.get('query16', ''),
                             query17=request.form.get('query17', ''),
                             query18=request.form.get('query18', ''),
                             query19=request.form.get('query19', ''))

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Files in same directory:")
    print("1. app.py")
    print("2. home.html")
    print("3. first_telc.csv (optional)")
    print("4. model.sav (required)")
    app.run(debug=True, port=5000)
