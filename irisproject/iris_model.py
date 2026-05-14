import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the data
df = pd.read_csv('Iris.csv')

# 2. Prepare Features and Target
# We drop 'Id' (irrelevant) and 'Species' (this is what we want to predict)
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

# 3. Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the Model
# We use 'max_iter' because Logistic Regression needs a few loops to "converge"
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Check Accuracy
predictions = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Save the model for deployment
joblib.dump(model, 'iris_logic_model.pkl')
print("Model saved as iris_logic_model.pkl")
