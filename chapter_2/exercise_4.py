import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv("../data/weight-height.csv")
plt.boxplot(df['Height'])
