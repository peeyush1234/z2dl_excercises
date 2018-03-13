import pandas as pd
import numpy as np

df = pd.read_csv("../data/weight-height.csv")
colors = np.where(df['Gender'] == 'Male', 'r', 'k')
df.plot(kind='scatter', x='Height', y='Weight', c=colors)
