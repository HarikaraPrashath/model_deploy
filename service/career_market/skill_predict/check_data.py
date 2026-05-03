import pandas as pd
import json

df = pd.read_csv('may3.csv')
print(f"Columns: {df.columns.tolist()}")
print(f"Unique Keywords: {df['keyword'].unique().tolist()}")
print(f"Data Head:\n{df.head()}")
