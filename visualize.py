import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Fertilizer Prediction.csv")

data.rename(columns={'Humidity ':'Humidity'}, inplace=True)

# Fertilizer Distribution
sns.countplot(x='Fertilizer Name', data=data)
plt.title("Fertilizer Distribution")
plt.show()

# Soil Type vs Fertilizer
sns.countplot(x='Soil Type', hue='Fertilizer Name', data=data)
plt.title("Soil Type vs Fertilizer")
plt.show()

# Crop Type vs Fertilizer
sns.countplot(x='Crop Type', hue='Fertilizer Name', data=data)
plt.title("Crop Type vs Fertilizer")
plt.show()

# Nutrient Analysis
sns.lineplot(data=data[['Nitrogen','Potassium','Phosphorous']])
plt.title("Nutrient Analysis")
plt.show()