# Linear Regression Model file for sample data Palmer's Penguins
# @author Ryan Magnuson rmagnuson@westmont.edu

# Setup
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
from sklearn import linear_model
from scipy.stats import pearsonr

# Data Reading
dataset = pd.read_csv("penguins.csv")
dataset = dataset.drop("sex", axis=1)
dataset = dataset.drop("rowid", axis = 1)
print(dataset.head())

# Plot
# plt.figure(figsize=(10,6))
# sns.displot(data=dataset.isna().melt(value_name="missing"), y="variable", hue="missing", multiple="fill")
# plt.savefig("penguin_stats.png")
# plt.close()

# Split data into training and testing
np.random.seed(1128)
train, test = train_test_split(dataset,test_size=.20)

def prep_data(data):
  ##
  # Takes data input and splits it into target/predictor vars
  # @param data: dataset
  # @return target/predictor vars

  df = data.copy()
  predictor = df.drop(["species"], axis=1) # "X"
  target = df["species"] # "y"

  return (predictor,target)

train = train.dropna()
test = test.dropna()

# split testing/training into predictor/target vars
X_train, y_train = prep_data(train)
X_test, y_test = prep_data(test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(dataset["species"].unique(), dataset["island"].unique())

num_vars = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year'] # 'rowid',
cat_vars = ['species', 'island']

# Density Plot
def density_plot(data, m_cols, m_rows):
  ##
  # Creates density plots for our numerical variables
  # @param data dataset to be used
  # @param m_cols number of columns
  # @param m_rows number of rows
  # @return density plot for different species base on numerical vars

  fig, ax = plt.subplots(m_rows, m_cols, figsize=(10,10))

  for i in range(len(num_vars)):
    var = data[num_vars[i]]
    row = i // m_cols
    col = i % m_cols

    sns.kdeplot(x=var, hue=train["species"], fill=True, ax=ax[row,col])
    plt.tight_layout() #fix spacing

# density_plot(train, 2, 3)
# plt.savefig("density_plot.png")
# plt.close()

# Correlation Plot (Matrix)
corrMatrix = train[num_vars].corr()
sns.heatmap(corrMatrix, annot=True)
plt.savefig("correlation_matrix.png")
plt.close()

# Print p-value
def p_val(var1, var2):
  ##
  # Computes correlation coefficients and corresponding p-vals
  # @param var1 var
  # @param var2 list of vars
  # @return printed correlation coefficient and p-val

  for element in var2:
    print("CC and p-val between", var1, "and", element, "is:\n", pearsonr(train[var1], train[element]))

penguin_stats = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
p_val("flipper_length_mm", penguin_stats)

le = preprocessing.LabelEncoder() #label encoder object

X_train["island"] = le.fit_transform(X_train["island"])
X_test["island"] = le.fit_transform(X_test["island"])

# Linear Regression Model (LRM)
LRM = linear_model.LinearRegression()
# LRM.fit(X_train, y_train) # ERROR HERE
#
# print('Coefficients: ', LRM.coef_)
# print('Variance score: {}'.format(LRM.score(X_test, y_test)))