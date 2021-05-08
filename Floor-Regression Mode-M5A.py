#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.simplefilter('ignore')
#plt.style.use('dark_background')


# In[2]:


data = pd.read_csv('Transformed_Housing_Data2.csv')
data.head()


# # Task:
# 1. To create a mean regression model based on "No of Floors" column and call it "floor_mean"
# 2. To compare the residual plots of overall "mean_sales" and "floor_mean".
# 3. To calculate the R-Square value for "floor_mean" model manually without using sklearn.

# ## 1. To create "floor_mean" column

# In[20]:


data['mean_sales'] = data['Sale_Price'].mean()
data['mean_sales'].head()


# In[21]:


## To check the unique values in column "No of Floors"

###### Start code ######

###### End code ######
data['No of Floors'].unique()


# ### Expected Output
# 
# 
# ```
# array([1. , 2. , 1.5, 3. , 2.5, 3.5])
# ```
# 

# In[22]:


## Using pandas.pivot_table() to calculate the "floor_mean"
### Start code ###
floors_mean = None


floors_mean=df.pivot_table(values='Sale_Price',columns='No of Floors',aggfunc=np.mean)
floors_mean


# ### Expected Output
# <img src="images/image1.png">

# In[23]:


# making new column
data['floor_mean'] = 0

# for every unique floor_mean, fill its mean price in new column "floor_mean"
for i in floor_mean.columns:
  ### start code ###


 data['floor_mean'][data['No of Floors']== i]=floors_mean[i][0]
                    
                    
                    ### end code ###

data['floor_mean'].head()


# ## 2. To Compare Residual plots

# ### Expected Output
# <img src="images/image2.png">

# In[24]:


data['mean_sales']


# In[26]:



## Calculating residuals floor_mean_difference and mean_difference
### start code###
mean_difference=0
floor_mean_difference=0
mean_difference = data['mean_sales']-data['Sale_Price']
floor_mean_difference = data['floor_mean']-data['Sale_Price']
### end code ###
mean_difference.size, floor_mean_difference.size


# ### Expected Outcome
# <img src="images/image3.png">

# In[29]:


## Plotting the Residuals for comparison

k = range(0, len(data)) # for x axis
l = [0 for i in range(len(data))] # for regression line in residual plot

plt.figure( figsize = (15,6), dpi =100)

################## plot for Overall Mean ####################
plt.subplot(1,2,1)
plt.scatter(k,mean_difference,color='red',label='Residuals',s=2)
plt.plot(k,l,color='green',label='mean Regression',linewidth=3)
plt.xlabel('Fitted points')
plt.ylabel('Residuals wrt floor mean')
#code to create the residual of mean regression model along with regression line
### start code ###
### end code ###
plt.title('Residuals with respect to no of floors')


################## plot for Overall Mean ####################
plt.subplot(1,2,2)
#code to create the residual of floor mean regression model along with regression line
### start code ###
plt.scatter(k,floor_mean_difference,color='red',label='Residuals',s=2)
plt.plot(k,l,color='green',label='mean Regression',linewidth=3)
plt.xlabel('Fitted points')
plt.ylabel("Residuals")

plt.title("Residuals with respect to overall mean")

plt.legend()
plt.show()


# ### Expected Outcome
# <img src="images/image4.png">

# ## 3. To calculate $R^2$ value of the "floor_mean" model manually
# <img src="images/image5.png">

# In[31]:


## Calculate mean square error for overall mean regression model and call it MSE 1
from sklearn.metrics import mean_squared_error

y=data['Sale_Price']
yhat1=data['mean_sales']
yhat2=data['floor_mean']

### start code ###
MSE1 = mean_squared_error(yhat1,y)
### end code ###

## Calculate mean square error for floor mean regression model and call it MSE 2
### start code ###
MSE2 = mean_squared_error(yhat2,y)
### end code ###

## calculate R-Square value using the formula and call it R2
### start code ###
R2 = 1-(MSE2/MSE1)
### end code ###
R2


# ### Expected Outcome
# <img src="images/image6.png">

# In[ ]:




