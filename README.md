# Customer Churn Prediction Web Application

This project uses Random Forest Classifier with SMOTEENN (Synthetic Minority Oversampling Technique + Edited Nearest Neighbors) (also Decision tree classifier with Smote) to handle imbalanced data and achieve high prediction accuracy.

##  Features

- **Real-time Predictions**: Instant churn probability predictions with confidence scores
- **Machine Learning Model**: Random Forest Classifier trained on balanced data using SMOTEENN
- **Interactive Web Interface**: User-friendly form to input customer data

## 📊 Model Performance

- **Accuracy**: 94%
- **Algorithm**: Random Forest Classifier with SMOTEENN
- **Data Balancing**: SMOTEENN technique to handle imbalanced dataset (73:27 ratio)
- **Features**: 19 customer attributes including demographics, services, and billing information

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, Random Forest Classifier
- **Data Processing**: pandas, numpy
- **Data Balancing**: imbalanced-learn (SMOTEENN)
- **Frontend**: HTML, CSS, Bootstrap 4

## 📁 Project Structure

```
CustomerChurnAnalysis/
├── app.py                 # Flask web application
├── home.html             # Frontend template
├── model.sav             # Trained Random Forest model (pickle file)
├── first_telc.csv        # Training dataset (optional)
├── churn_analysis.py     # EDA and model training notebook
├── model_training.py     # Model training script
└── README.md            # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/aarchiie/Customer-Churn-Analysis-Prediction
cd CustomerChurnAnalysis
```

### 2. Install Required Packages
```bash
pip install pandas scikit-learn flask pickle numpy imbalanced-learn
```

### 3. Verify Required Files
Ensure these files are in your project directory:
- `app.py` - Main Flask application
- `home.html` - Frontend template
- `model.sav` - Trained model file
- `first_telc.csv` - Training data (optional)

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## 📋 Input Features

The application requires the following customer information:

### Customer Demographics
- **Senior Citizen**: Whether the customer is a senior citizen (Yes/No)
- **Gender**: Customer's gender (Male/Female)
- **Partner**: Whether the customer has a partner (Yes/No)
- **Dependents**: Whether the customer has dependents (Yes/No)

### Account Information
- **Tenure**: Number of months the customer has stayed (1-72)
- **Monthly Charges**: Monthly billing amount ($)
- **Total Charges**: Total amount charged ($)
- **Contract**: Contract type (Month-to-month/One year/Two year)

### Services
- **Phone Service**: Phone service subscription (Yes/No)
- **Multiple Lines**: Multiple phone lines (Yes/No/No phone service)
- **Internet Service**: Internet service type (No/DSL/Fiber optic)
- **Online Security**: Online security service (Yes/No/No internet service)
- **Online Backup**: Online backup service (Yes/No/No internet service)
- **Device Protection**: Device protection service (Yes/No/No internet service)
- **Tech Support**: Technical support service (Yes/No/No internet service)
- **Streaming TV**: Streaming TV service (Yes/No/No internet service)
- **Streaming Movies**: Streaming movies service (Yes/No/No internet service)

### Billing & Payment
- **Paperless Billing**: Paperless billing preference (Yes/No)
- **Payment Method**: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)

## 🤖 Model Details

### Algorithm: Random Forest Classifier
- **Estimators**: 100 trees
- **Criterion**: Gini impurity
- **Max Depth**: 6
- **Min Samples Leaf**: 8
- **Random State**: 100

### Data Preprocessing
1. **Data Cleaning**: Handled missing values in TotalCharges column
2. **Feature Engineering**: Created tenure groups (1-12, 13-24, 25-36, etc.)
3. **Encoding**: One-hot encoding for categorical variables
4. **Balancing**: Applied SMOTEENN to handle class imbalance

### Key Insights from EDA
- **High Churn Factors**: Month-to-month contracts, no online security, no tech support, first year customers, fiber optic internet
- **Low Churn Factors**: Long-term contracts, no internet service, 5+ years tenure
- **Neutral Factors**: Gender, phone service, multiple lines have minimal impact

## 📈 Usage Example

1. **Fill the Form**: Enter customer details in the web form
2. **Submit**: Click "PREDICT CHURN" button
3. **View Results**: Get prediction result with confidence percentage

**Sample Output:**
- "This customer is likely to churn (leave)! Confidence: 85.3%"
- "This customer is likely to continue! Confidence: 92.1%"

## 🔍 Troubleshooting

### Common Issues

1. **Template Not Found Error**
   ```
   Solution: Ensure home.html is in the same directory as app.py
   ```

2. **Model File Not Found**
   ```
   Solution: Ensure model.sav exists in the project directory
   ```

3. **Import Errors**
   ```
   Solution: Install missing packages using pip install <package-name>
   ```

4. **Invalid Input Error**
   ```
   Solution: Ensure numeric fields contain valid numbers
   ```

## 📊 Model Training

To retrain the model with new data:

1. **Prepare Data**: Update the training dataset
2. **Run Training Script**: Execute the model training pipeline
3. **Data Processing**: 
   - Handle missing values
   - Create feature groups
   - Apply one-hot encoding
   - Balance data using SMOTEENN
4. **Train Model**: Fit Random Forest Classifier
5. **Save Model**: Export trained model as pickle file
