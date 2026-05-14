# STEP 1: Import the tools we need
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# STEP 2: Load your data
# This reads the file you provided 
df = pd.read_csv('Iris.csv')

# STEP 3: Clean the data
# We remove 'Id' because it's just a row number and doesn't help identify flowers 
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] 
y = df['Species'] # This is what we want to predict: Setosa, Versicolor, or Virginica 

# STEP 4: Split the data
# We give 80% to the model to study, and keep 20% to test it later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Create the "Brain" (The Model)
# We use K-Nearest Neighbors - it looks at similar flower measurements to guess the species
model = KNeighborsClassifier(n_neighbors=3)

# STEP 6: Training
# The model "studies" the training data
model.fit(X_train, y_train)

# STEP 7: Testing
# We ask the model to guess the species for the 20% we held back
predictions = model.predict(X_test)

# STEP 8: Check the Score
# Compare the model's guesses to the real answers
score = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {score * 100}%")

# BONUS: Try it with your own custom flower!
# Let's pretend we found a flower with these measurements:
new_flower = [[5.1, 3.5, 1.4, 0.2]] # These look like Setosa numbers! 
prediction = model.predict(new_flower)
print(f"The model says this new flower is: {prediction[0]}")