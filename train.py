import pandas as pd

df = pd.read_csv("dataset/online_shoppers_intention.csv")

print(df.isnull().sum())