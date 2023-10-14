# ODD2023-Datascience-Ex06
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.
# ALGORITHM
1. Read the given Data
2. Perform Data cleaning process on the dataset.
3. Apply Feature Transformation techniques to all the features of the data set
4. Analyse the transformed features
# CODE AND OUTPUT
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/aedac24e-c5dc-4b96-8b4a-8f2bab84594d)

```
df.info()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/d002eb70-ae46-47fa-903d-0d976ba9610c)

```
df.skew()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/c72dd081-7edc-4175-9c8a-3f09c5e69c85)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/1c1466a3-2371-427a-9453-4d29df617da2)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/99b3b046-74c9-4ebe-85b0-88519e9e619d)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/d2af263b-885e-4b05-aef4-3ff6ea6a346e)

```
np.square(df['Highly Positive Skew'])
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/e5117342-5045-41ab-a940-2a852429ba21)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/b87a2380-5eda-4931-93ad-a07323f42805)

```
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/155e1e88-ae20-42e9-9886-4a5f217b6e46)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[['Moderate Negative Skew']])
df
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/52453472-ea35-4a76-a220-76bab9f7d192)

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
```
```
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/a8c6beb5-60fc-44ea-bccd-adeca59312d9)

```
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/9c19d20a-6038-4c11-960c-62239f28fa13)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[['Highly Negative Skew']])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/589a6c8d-a0a7-4aea-b1f0-9a0615e78621)

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex06/assets/118904526/c319964c-cdea-4739-af3b-f6fee56f3039)
