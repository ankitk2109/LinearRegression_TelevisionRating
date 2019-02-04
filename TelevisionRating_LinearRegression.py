
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd

#Reading Data
dataset = pd.read_csv("DS_Usecase.csv",encoding = "ISO-8859-1")
X = dataset.iloc[:,[4,6,7,8]].values

X_plot = dataset.iloc[:,[4,6]].values

Y = dataset.iloc[:,-1].values


#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer( missing_values = "NaN", strategy = "mean", axis = 0 )
Y = Y.reshape(-1,1)
imputer = imputer.fit(Y)
Y = imputer.transform(Y)

#Transformation
dataset.describe()

min = Y.min()
max = Y.max()
avg = Y.mean()
avg = np.mean(Y, axis = 0)
med = np.median(Y,axis = 0)
percentile_25 = np.percentile(Y,25,axis= 0)
percentile_75 = np.percentile(Y,75,axis= 0)
std_dev = np.std(Y, axis = 0)
skew_var = sc.stats.skew(Y, axis=0)

#Encoding Categorial Data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder ()
X[:,2] = labelencoder_X.fit_transform(X[:,2])
'''X_fit = labelencoder_X.fit(X[:,2])
X[:,2] = X_fit.transform(X[:,2])'''
oneHotEncoder_X = OneHotEncoder(categorical_features=[2])
X = oneHotEncoder_X.fit_transform(X).toarray()

#Avoide Dummy Trap
X = X[:,1:]

#Visualising Data
plt.hist(Y, bins=30)
plt.scatter(X[:,2], Y)

#Spaghetti  plot
df2=pd.DataFrame(data=X_plot, columns=["rating_level", "user_rating_size"])
# style
plt.style.use('seaborn-darkgrid')
# create a color palette
palette = plt.get_cmap('Set1')
# multiple line plot
num=0
for column in df2:
    num+=1
    plt.plot(df2[column],color=palette(num), linewidth=4,alpha=0.9)

# Adding legend
plt.legend(loc=2, ncol=2)

# Adding titles
plt.title("A Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("rating_level")
plt.ylabel("user_rating_size")









































