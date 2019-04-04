
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import os


# In[187]:


# file paths
string_path = os.path.abspath('./')
string_file_train = 'train_technidus_clf.csv'
string_file_test = 'test_technidus_clf.csv'
string_fp_train = os.path.join(string_path, string_file_train)
string_fp_test = os.path.join(string_path, string_file_test)

#Read CSV files into python
Df_1 = pd.read_csv(string_fp_train)
Df_2 = pd.read_csv(string_fp_test)

Df_1.head()

Df_2.info()


# In[188]:


#Merge Training and Testing dataset

df = pd.concat([Df_1, Df_2],ignore_index=True)


# In[189]:


#Perfrom Exploratory Data Analysis.
df.info()


# In[190]:


df.head()


# In[192]:


#Drop dimensions which are not useful to analysis

data=df.drop(["AddressLine1", "AddressLine2", "BirthDate", "CustomerID", "FirstName","StateProvinceName","PhoneNumber", "Suffix", "Title","LastName", "MiddleName","NumberChildrenAtHome", "City", "PostalCode"], axis=1)
data.head()


# In[193]:


#Save Clean data

data.to_csv ("C:/Users/Sdeol/Desktop/all/bikebuyer2.csv")


# In[194]:



df= pd.read_csv ("C:/Users/Sdeol/Desktop/all/bikebuyer2.csv")


# In[195]:


df.info()


# In[196]:


df.head()


# In[199]:


#Separate into dependent and independent variables

X = df.iloc[:,:-1].values
y = df.iloc[:,2].values
print(y)


# In[200]:


#spliting data columns into numeric and categorical

df_num_counter=0
df_other_counter=0
for col in df:
    if (df[col].dtype)in ["int64", "float64"]:
        df_num_col=pd.DataFrame(df[col], columns=[col])
        if df_num_counter==0:
            df_num=df_num_col
        else:
            df_num=df_num.join(df_num_col)
        df_num_counter=df_num_counter+1
    else:
        df_other_col=pd.DataFrame(df[col], columns=[col])
        if df_other_counter==0:
            df_other=df_other_col
        else:
            df_other=df_other.join(df_other_col)
        df_other_counter=df_other_counter+1 


# In[201]:


#reading numerical data

df_num.head()


# In[113]:


#converting CountryRegionName to a categorical column for future convenience

df["CountryRegionName"]=df["CountryRegionName"].astype(str)
df["CountryRegionName"].dtype


# In[114]:


#converting Education to a categorical column for future convenience

df["Education"]=df["Education"].astype(str)
df["Education"].dtype


# In[220]:


#converting Education to a categorical column for future convenience

df["Occupation"]=df["Occupation"].astype(str)
df["Occupation"].dtype


# In[221]:


df.head()


# In[222]:


#displaying non-numeric columns data frame
df_other.head()


# In[223]:


df_num.isnull().sum()


# In[224]:


#encoding non_numeric columns

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
i=0
for col in df_other:
    x_enc=encoder.fit_transform(df_other[col])
    df_enc=pd.DataFrame(x_enc, columns=[col])
    if i>0:
        df_cat=df_cat.join(df_enc)
    else:
        df_cat=df_enc
    i=i+1
df_cat.head()


# In[225]:


#splitting encoded columns into binary and multi-category data frames

bin_count=0
mult_count=0

for col in df_cat:
    #print(df_cat[col].value_counts())
    if len(df_cat[col].unique())>2:
        #print (col, "has ", len(df_cat[col].unique()), " unique values")
        mult_col=df_cat[col]
        if mult_count==0:
            df_mult=pd.DataFrame(mult_col, columns=[col])
        else:
            df_mult=df_mult.join(mult_col)
        mult_count=mult_count+1
    else:
        bin_col=df_cat[col]
        if bin_count==0:
            df_bin=pd.DataFrame(bin_col, columns=[col])
        else:
            df_bin=df_bin.join(bin_col)
        bin_count=bin_count+1


# In[226]:


#displaying multiple category columns
df_mult.head()


# In[227]:


#displaying binary columns
df_bin.head()


# In[239]:


#combinig one_hot encoded and binary columns

df_complete=df_num.join(df_cat)
df_complete.head()


# In[240]:


#Filling missing values with median score

median_value=df_complete['BikeBuyer'].median()
df_complete['BikeBuyer']=df_complete['BikeBuyer'].fillna(median_value)


# In[241]:


#saving data to a file
df_complete.to_csv('C:/Users/Sdeol/Desktop/all/bicyclefinal.csv')


# In[242]:


#Depedendent and Independent variables

X = df_complete.iloc[:,:-1].values
y = df_complete.iloc[:,2].values
print(y)


# In[243]:


import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC


# In[244]:


#Splitting dataset
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)


# In[245]:


from sklearn.linear_model import LogisticRegression


# In[246]:


#Import and fit LogisticRegression classifier
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train, y_train.ravel())


# In[247]:


#assessing model accuracy for train dataset using cross-validation

y_train_pred=log_reg.predict(x_train)
y_test_pred=log_reg.predict(x_test)
from sklearn.model_selection import cross_val_score
cross_val_score(log_reg, x_train, y_train.ravel(), cv=20, scoring="accuracy")


# In[248]:


print (y_test_pred)


# In[249]:


import numpy as np
import pandas as pd
prediction = pd.DataFrame(y_test_pred, columns=['predictions']).to_csv('Logreg.csv')


# In[250]:


y_test_pred=log_reg.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test_pred, y_test.ravel())
print (accuracy)


# In[251]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

accuracy_score(y_test, y_predict)

print (y_predict)


# In[252]:


import numpy as np
import pandas as pd
submission2 = pd.DataFrame(y_predict, columns=['BikeBuyer']).to_csv('Submission2.csv')


# In[253]:


nccf=NeverChurnClassifier()
y_train_nccf_pred=nccf.predict(x_train)
n_correct=sum(y_train_nccf_pred==y_train)
n_correct
print("Baseline Accuracy: ", n_correct/len(y_train_pred))


# In[254]:


#import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(x_train, y_train).predict(x_test)
#print(pred.tolist())

#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))


# In[255]:


print(pred)


# In[256]:


import numpy as np
import pandas as pd
Submission3 = pd.DataFrame(pred, columns=['BikeBuyer']).to_csv('Gaussian.csv')


# In[257]:


#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#create an object of type LinearSVC
svc_model = LinearSVC(random_state=42)

#train the algorithm on training data and predict using the testing data
pred2 = svc_model.fit(x_train, y_train).predict(x_test)

#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(y_test, pred2, normalize = True))


# In[258]:


print (pred2)


# In[182]:


import numpy as np
import pandas as pd
prediction = pd.DataFrame(pred2, columns=['BikeBuyer']).to_csv('SVC.csv')


# In[260]:


#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)

#Train the algorithm
neigh.fit(x_train, y_train)

# predict the response
pred1 = neigh.predict(x_test)

# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred1))


# In[261]:


print (pred1)


# In[183]:


import numpy as np
import pandas as pd
prediction = pd.DataFrame(pred1, columns=['BikeBuyer']).to_csv('Knearest.csv')

