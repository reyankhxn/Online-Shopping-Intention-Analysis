import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/online_shoppers_intention.csv")

df["Revenue"].value_counts().plot(kind="bar")
plt.title("Purchase Distribution")
plt.show()