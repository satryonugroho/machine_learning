import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pd.read_csv('history_customers.csv')

data = pd.get_dummies(data, columns=['GENDER', 'EDUCATION_LEVEL', 'MARITAL_STATUS', 'INCOME_CATEGORY', 'CARD_CATEGORY'])

X = data.drop(['STATUS'], axis=1)
y = data['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Attrited Customer')
recall = recall_score(y_test, y_pred, pos_label='Attrited Customer')

print(f"Accuracy = {accuracy}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]
feature_names = X.columns

# plt.figure(figsize=(10, 6))
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices], align="center")
# plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.show()

new_data = pd.read_csv('new_customer.csv')
new_data = pd.get_dummies(new_data, columns=['GENDER', 'EDUCATION_LEVEL', 'MARITAL_STATUS', 'INCOME_CATEGORY', 'CARD_CATEGORY'])
new_data_scaled = scaler.transform(new_data)
new_data_pred = model.predict(new_data_scaled)
# print(f"new data = {new_data_pred}")

new_data['PREDICTION'] = new_data_pred
new_data.to_csv('export\_new_data_with_predictions.csv', index=False)
joblib.dump(model, 'random_forest_model.pkl')