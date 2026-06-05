import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/online_shoppers_intention.csv")

df["VisitorType"].value_counts().plot(kind="bar")

plt.title("Visitor Type Distribution")
plt.show()