import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data = pd.read_csv('road_accident_data.csv')

# Select relevant features and target column
features = [
    'Day_of_Week', 'Junction_Control', 'Junction_Detail', 'Latitude', 'Longitude',
    'Light_Conditions', 'Local_Authority_(District)', 'Carriageway_Hazards', 
    'Number_of_Casualties', 'Number_of_Vehicles', 'Police_Force', 
    'Road_Surface_Conditions', 'Road_Type', 'Speed_limit', 'Time', 
    'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type'
]
X = data[features]
y = data['Accident_Severity']

# Encode categorical features and target
X = pd.get_dummies(X, drop_first=True)
y = y.map({'Slight': 0, 'Serious': 1, 'Fatal': 2, 'Fetal': 3})

# Split the dataset into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the logistic regression model with class weights
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Save the trained model
joblib.dump(model, 'accident_severity_model.pkl')

print("Model training completed and saved as 'accident_severity_model.pkl'.")
