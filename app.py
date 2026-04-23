from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = Flask(__name__)

# -----------------------------
# LOAD DATA & TRAIN MODELS ONCE
# -----------------------------
data = pd.read_csv("cleaned_data.csv", header=None)

data.columns = [
    'Temparature',
    'Humidity',
    'Moisture',
    'Soil Type',
    'Crop Type',
    'Nitrogen',
    'Potassium',
    'Phosphorous'
]

X = data[['Temparature','Humidity','Moisture',
          'Soil Type','Crop Type',
          'Nitrogen','Potassium','Phosphorous']]

y = data['Fertilizer Name']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name


# -----------------------------
# ROUTES
# -----------------------------

@app.route('/')
def home():
    return render_template("index.html",
                           acc=round(best_accuracy * 100, 2),
                           model_name=best_model_name)


@app.route('/predict', methods=['POST'])
def predict():

    temp = int(request.form['temp'])
    hum = int(request.form['hum'])
    moist = int(request.form['moist'])
    soil = int(request.form['soil'])
    crop = int(request.form['crop'])
    n = int(request.form['nitrogen'])
    k = int(request.form['potassium'])
    p = int(request.form['phosphorous'])

    input_data = np.array([[temp, hum, moist, soil, crop, n, k, p]])
    input_data = scaler.transform(input_data)

    prediction = best_model.predict(input_data)

    return render_template("result.html",
                           pred=prediction[0],
                           model_name=best_model_name,
                           acc=round(best_accuracy * 100, 2))


if __name__ == "__main__":
    app.run(debug=True)
