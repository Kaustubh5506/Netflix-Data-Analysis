# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv(r'C:\Users\ADMIN\Downloads\mymoviedb.csv', lineterminator='\n')

# Displaying the first 3 rows
print(df.head(3))

# Getting DataFrame info
print(df.info())

# Checking for duplicate rows
print(df.duplicated().sum())

# Converting 'Release_Date' column to datetime
df['Release_Date'] = pd.to_datetime(df['Release_Date'])
print(df)

# Checking the datatype of 'Release_Date'
print(df['Release_Date'].dtype)

# Extracting only the year from 'Release_Date'
df['Release_Date'] = df['Release_Date'].dt.year
print(df)

# Dropping unnecessary columns
cols = ['Original_Language', 'Poster_Url', 'Overview']
df.drop(cols, axis=1, inplace=True)

# Displaying first 3 rows after dropping columns
print(df.head(3))

# Splitting 'Genre' column by comma and exploding it into separate rows
df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre').reset_index(drop=True)

# Displaying the modified DataFrame
print(df.head(3))

# Defining a function to categorize a column based on quantile edges
def categories_cols(df, cols, labels):
    edges = [df[cols].describe().loc['min'],
             df[cols].describe().loc['25%'],
             df[cols].describe().loc['50%'],
             df[cols].describe().loc['75%'],
             df[cols].describe().loc['max']]

    df[cols] = pd.cut(df[cols], edges, labels=labels, duplicates='drop')
    return

# Label definitions for Vote_Average categorization
labels = ['non_popular', 'below_Average', 'Average', 'Popular']

# Applying categorization on 'Vote_Average'
categories_cols(df, 'Vote_Average', labels)

# Displaying unique categories
print(df['Vote_Average'].unique())
print(df.head(3))

# Displaying value counts of each category
print(df['Vote_Average'].value_counts())

# Dropping any rows with missing values
print(df.dropna(inplace=True))

# Checking for null values
print(df.isna)

# Setting plot style
sns.set_style('whitegrid')

# Plotting genre distribution
sns.catplot(y='Genre', data=df, kind='count', color="#de1717", order=df['Genre'].value_counts().index)
plt.title('Genre column distribution')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Plotting Vote_Average distribution
sns.catplot(y='Vote_Average', data=df, kind='count', color="#0ed215ff", order=df['Vote_Average'].value_counts().index)
plt.title('Average distribution')
plt.xlabel('Count')
plt.ylabel('Vote_Average')
plt.show()

# Displaying movie(s) with minimum popularity
print(df[df['Popularity'] == df['Popularity'].min()])

# Displaying movie(s) with maximum popularity
print(df[df['Popularity'] == df['Popularity'].max()])

# Plotting Release_Date histogram
df['Release_Date'].hist()
plt.title('Release_Date  distribution')
plt.xlabel('Count')
plt.ylabel('Release_Date')
plt.show()
