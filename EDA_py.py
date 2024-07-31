
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pickle, joblib
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_validate

# Loading the data set
data = pd.read_csv("C:/Users/ADMIN/Desktop/Project/machine_downtime_2.csv", encoding = "ISO-8859-1")

# Mapping the type to numeric values 1 and 0. 
# This step is required for metric calculations in the model evaluation phase.



###############################################################################
# PostgreSQL
# pip install psycopg2 
import psycopg2 # Psycopg2 is a PostgreSQL database driver, it is used to perform operations on PostgreSQL using Python, it is designed for multi-threaded applications
from sqlalchemy import create_engine
  
# Creating an engine that connects to PostgreSQL
# conn_string = psycopg2.connect(database = "postgres", user = 'postgres', password = 'monish1234', host = 'localhost', port= '5432')


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "prasanna", # passwrd
                               db = "machine_downtime")) #database

data.to_sql('machine_downtime', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


#### Read the Table (data) from MySQL database

sql = 'SELECT * FROM machine_downtime'
# sql2="show tables"
# tables = pd.read_sql_query(sql2, engine)

dataset = pd.read_sql_query(sql, engine)

dataset.head()

dataset.describe()


dataset.info()

dataset.shape

dataset = dataset.drop(columns=['Date','Assembly_Line_No', 'Machine_ID'])

# Predictors
X = dataset.iloc[:,0:14]
X

# Target
Y = dataset.iloc[:,-1]
Y.shape


# Seperating Integer and Float data

numeric_features = X.select_dtypes(include = ['float64']).columns
numeric_features.shape



# Segregating categorical data based on their data type
cat_features = X.select_dtypes(include = ['object']).columns
cat_features

############################ Data analysis #############################################

# X['Machine_ID'].value_counts()
# X['Assembly_Line_No'].value_counts()

# dataset['Downtime'].value_counts()

X.isnull().sum()

########################################  Missing values Analysis ##############################################################

# Define pipeline for missing data if any

# Assuming numeric_features is a list of the remaining numeric column names in your X DataFrame
num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])

preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, numeric_features)])


# Fit the imputation pipeline to input features (X)
X_numeric = X[numeric_features]  # Select only the numeric columns
imputation = preprocessor.fit(X_numeric)  # Fit to the numeric columns in X

# Save the pipeline
joblib.dump(imputation, 'medianimpute.pkl')

# Transformed data
cleandata = pd.DataFrame(imputation.transform(X_numeric), columns=numeric_features)
cleandata


cleandata.isnull().sum()

cleandata.duplicated().sum()



# Boxplot
cleandata.plot(kind = 'box', subplots = True, sharey = False, figsize = (30, 15)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# Plot histograms for each column in cleandata before scaling
plt.figure(figsize=(20, 10))
for i, col in enumerate(cleandata.columns, 1):
    plt.subplot(5, 3, i)  # Adjust the subplot grid as per your number of columns
    plt.hist(cleandata[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


##################################################### Scaling ######################################################################
## Scaling with robustScaler

from sklearn.preprocessing import RobustScaler

scale_pipeline = Pipeline([('scale', RobustScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) # Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata)

# Save Minmax scaler pipeline model
joblib.dump(scale, 'robust_scaler.pkl')

scaled_data = pd.DataFrame(scale.transform(cleandata), columns = numeric_features)
scaled_data.describe()
scaled_data.to_csv('scaled_data')

# Plot histograms for each column for scaled data 

plt.figure(figsize=(20, 10))
for i, col in enumerate(scaled_data.columns, 1):
    plt.subplot(5, 3, i)  # Adjust the subplot grid as per your number of columns
    plt.hist(scaled_data[col], bins=20, color='yellow', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel('Value', fontweight='bold')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Calculate skewness for each numeric column
skewness = scaled_data.skew()

# Create a KDE plot for skewness
sns.kdeplot(skewness, shade=True)
plt.title('KDE Plot')
plt.xlabel('Skewness')
plt.ylabel('Density')
plt.show()


# Plot the distplot for skewness

sns.set(style="whitegrid")  # Set the style
sns.distplot(scaled_data , kde=True, color="skyblue", hist_kws={"linewidth": 15, 'alpha': 1})
plt.show()




#############################################################  Encoding ##################################################################
# Categorical features
# encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])

# preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, cat_features)])

# clean =  preprocess_pipeline.fit(X)   # Works with categorical features only

# # Save the encoding model
# joblib.dump(clean, 'encoding')

# encode_data = pd.DataFrame(clean.transform(X))


# # To get feature names for Categorical columns after Onehotencoding 
# encode_data.columns = clean.get_feature_names_out(input_features = X.columns)
# encode_data.info()


# clean_data = pd.concat([scaled_data, encode_data], axis = 1)  # concatenated data will have new sequential index
# clean_data.info()
# clean_data.describe().T  # Transposed view


# # Assuming Y_encoded is your NumPy array
# clean_data_df = pd.DataFrame(clean_data)
# clean_data_df.to_csv('clean_data.csv', index=False)


############################################################# EDA ###########################################################
import sweetviz as sv

# Analyzing the dataset
report = sv.analyze(scaled_data)

# Display the report
# report.show_notebook()  # integrated report in notebook

report.show_html('EDAreport.html') # html report generated in working directory

dataplot = sns.heatmap(scaled_data.corr(), annot = True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})

dataplot


# Assuming `clean_data` is your cleaned DataFrame containing only numeric columns
sns.pairplot(scaled_data)
plt.show()

################################## Encoding for target column #################################################################################################

# Assuming df is your DataFrame
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)


# Assuming Y_encoded is your NumPy array
Y_encoded_df = pd.DataFrame(Y_encoded)
Y_encoded_df.to_csv('encoded_data.csv', index=False)

######################################################### BUSINESS MOMENTS ###############################################################################

# Calculating mode for categorical features


dataset['Downtime'].mode()

dataset['Downtime'].value_counts()

# dataset['Machine_ID'].mode()

# dataset['Machine_ID'].value_counts()

# dataset['Assembly_Line_No'].mode()


#Calculating all four business moments for scaled data 

# Calculate mean for each column
mean_values = scaled_data.mean()

print("Mean values:")
print(mean_values)


# Calculate variance for each column
variance_values = scaled_data.var()


print("\nVariance values:")
print(variance_values)


# Calculate standard deviation for each column
std_deviation_values = scaled_data.std()

print("\nStandard deviation values:")
print(std_deviation_values)


#calculate skewness for each column

skewness = scaled_data.apply(skew)

print("\n skewness values:")
print(skewness)



# Calculate kurtosis for each column

kurtosis_values = scaled_data.apply(lambda x: x.kurtosis())

print("Kurtosis values:")
print(kurtosis_values)






