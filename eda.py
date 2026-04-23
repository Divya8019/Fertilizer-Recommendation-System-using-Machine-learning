import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Fertilizer Prediction.csv")

# Remove space in Humidity column name
data.rename(columns={'Humidity ':'Humidity'}, inplace=True)

# Encode text columns
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

data['Soil Type'] = le1.fit_transform(data['Soil Type'])
data['Crop Type'] = le2.fit_transform(data['Crop Type'])
data['Fertilizer Name'] = le3.fit_transform(data['Fertilizer Name'])

data = pd.read_csv("cleaned_data.csv", header=None)

print("Data Cleaned Successfully")
