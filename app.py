import pandas as pd

df = pd.read_csv("data/credit_fraud.csv")

print(df.head())
print(df.info())

print(df['is_fraud'].value_counts())

# handle missing value
df = df.dropna()

# encode categorical
df = pd.get_dummies(df)

# pisahkan fitur & label
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))