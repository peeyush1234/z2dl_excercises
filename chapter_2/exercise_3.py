import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv("../data/weight-height.csv")
plt.hist(df['Height'],alpha=0.6)
plt.axvline(df[df['Gender'] == 'Male']['Height'].mean())
plt.axvline(df[df['Gender'] == 'Female']['Height'].mean())
