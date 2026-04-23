import pandas as pd

df = pd.read_csv("data/credit_fraud.csv")

print(df.head())
print(df.info())

print(df['is_fraud'].value_counts())