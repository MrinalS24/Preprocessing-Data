# preparing raw data for machine learning

#1. data cleaning
# What if data is not clear or incomplete

#2. hadling missing values

#3. ecoding
#converting values, text to numbers, repeated texts

#4. feature scaling
#making sure the data has values of different scale and level
#if salary and age are analysed, different units but use with same level

#5. removing outliers
#data that does not represent the actual pattern of the data

#6. dividing our existing data into training and testing. 
#have some test data, that you can check how close your analysis is. take some data to train the machine

import pandas as pd
from sklearn.preprocessing import LabelEncoder
# sklearn is a very wide machine learning library
from sklearn.preprocessing import StandardScaler
 


sample_data_set = {
    "Age" : [23, 28, 30, 25, 33, 26, None, 34, None, None, 33],
    "Gender" : ["Male", "Female", None, "Female ", "Male", "Male", None, "Female", "Male ", "Female", "Male"], 
    "Salary" : [30, 20, None, 32, 20, 45, 34, 26, 33, 17, 68],
    "Department" : ["HR", " Finance", "IT", "Creative", "Manager ", "HR", "IT", None, "Creative ", "Finance", "CEO"],
    "Car" : ["No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(sample_data_set)
print(df)
print("----" * 45)


#starting data cleaning
# removing spaces from all text based columns
text_columns = df.select_dtypes(include = "object").columns

for col in text_columns:
    print(df[col])
    df[col] = df[col].str.strip()

print(df)
print("----" * 45)

# removing or missing values (None)
# for numbers or numeric values, find the mean
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
# for text values, find the mode
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

print(df)
print("----" * 45)


#encoding text data, with different values because machine learning cannot read
#or use the text data to process or analyse, so you replace the text with a certain number

encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])
df["Department"] = encoder.fit_transform(df["Department"])
df["Car"] = encoder.fit_transform(df["Car"])

print(df)
print("----" * 45)


# scale down and bring equivalence to two different columns to bring them at a same kind of level. 

scaler = StandardScaler()

df[["Age", "Salary"]] = scaler.fit_transform(df[["Age", "Salary"]])

print(df)
print("----" * 45)

#removing outliers, using quantiles
q1 = df["Salary"].quantile(0.25)
q3 = df["Salary"].quantile(0.75)
iqr = q3 - q1

df = df[(df["Salary"] >= q1 - 1.5*iqr) & (df["Salary"] <= q3 + 1.5 * iqr)]

print(df)
print("----" * 45)