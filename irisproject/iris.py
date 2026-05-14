import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Iris.csv')

# Drop the 'Id' column as it doesn't help with classification 
df = df.drop('Id', axis=1)

# Features (X) and Target (y)
X = df.drop('Species', axis=1) # SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm 
y = df['Species'] # Iris-setosa, Iris-versicolor, Iris-virginica