import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
import pandas as pd

def visualizetempweatherdata():
    df=pd.read_csv('car_sales.csv')
    df['Month']=pd.to_datetime(df['Month'])
    df.set_index('Month',inplace=True)
    df.plot()
    plt.title("temperature over time")
    plt.xlabel('Month')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()

visualizetempweatherdata()