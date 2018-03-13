import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/international-airline-passengers.csv")

#df.head()
df['Month'] = pd.to_datetime(df['Month'])

df.set_index('Month')

plt.plot(df['Month'], df['Thousand Passengers'])
plt.title('Line plot')
plt.legend(['data1', 'data2'])
plt.xlabel("Time")
plt.ylabel("Thousand Passengers")
