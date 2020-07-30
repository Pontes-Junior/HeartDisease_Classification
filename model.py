# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
data = pd.read_csv('heart.csv')


#Selecting Features and Target
features=['age', 'sex', 'cp', 'trestbps',"chol","fbs"]
X = data[features]
y = data.target


# Model Building
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500,max_depth=10,random_state=42)
model.fit(X_train, y_train)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[60,0,0,145,150,0]])) 