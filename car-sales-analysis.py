import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data= pd.read_csv('car_sales.csv')

#analyze cars data sales
data.head()
plt.scatter(data['Price_in_thousands'],data['Sales_in_thousands'])
plt.xlabel('Price')
plt.ylabel('Sales_in_thousands')
plt.title('Price vs Sales_in_thousands')
plt.show()
plt.bar(data['Manufacturer'],data['Sales_in_thousands'])
plt.xlabel('Manufacturer')  
plt.ylabel('Sales_in_thousands')
plt.title('Manufacturer vs Sales_in_thousands')
plt.show()
plt.pie(data['Manufacturer'],data['Sales_in_thousands'])
plt.xlabel('Manufacturer')  
plt.ylabel('Sales_in_thousands')
plt.title('Manufacturer vs Sales_in_thousands')
plt.show()

