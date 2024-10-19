import pandas as pd 
import matplotlib.pyplot as plt 


data = pd.read_csv('C:\\Users\\visse\\OneDrive\\Desktop\\2024\TIGB_Calls_status.csv')

print(data)
df = pd.DataFrame(data=data,columns=data.columns)

#fig = plt.figure(figsize = (10, 5))
#plt.bar(data=data,columns,data[::],color ='maroon')
#plt.show()
import numpy as np
