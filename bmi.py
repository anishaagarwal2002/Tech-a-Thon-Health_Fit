#!/usr/bin/env python
# coding: utf-8

# # TEAM NAME: JOSH
# # TEAM LEADER: ANISHA AGARWAL
# # EMAIL-ID: anishaagrawal2002@gmail.com
# # TEAM MEMBER: HIMAANGI MOYAL
# # EMAIL-ID: himaangi123@gmail.com

# In[ ]:


import csv
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as sk
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as nd
from sklearn import metrics
warnings.filterwarnings('ignore')


# In[6]:


# csv file name
filename = "dataset.csv"


# In[7]:


fields = []
rows = []


# ## Read the data

# In[8]:


with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))


# In[9]:


print(fields)


# In[10]:


fields_X = fields[:3]
fields_y = fields[-1:]


# In[11]:


rows_X = []
rows_y = []
for row in rows:
#     print(row[:3])
    if row[0] == 'Male' or row[0] == 0:
        row[0] = 0
    else:
        row[0] = 1
    rows_X.append(row[:3])
    rows_y.append(row[-1:])


# # Dividing dataset into test and train

# In[12]:


X_train, X_test, y_train,y_test = train_test_split(rows_X, rows_y, test_size=0.20, random_state=123, stratify=rows_y)


# # Data Preprocessing

# In[13]:


# from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Model Trained(Logistic Regression)

# In[14]:


# import sklearn.linear_model as sk
# from sklearn.model_selection import StratifiedKFold, cross_val_score


logis = sk.LogisticRegression(multi_class = 'multinomial', max_iter=2500, C = 1e5, class_weight = 'balanced', solver = 'lbfgs')
# print(logis)
clf = logis.fit(X_train, y_train)
# print(clf)

kf = StratifiedKFold(shuffle=True, n_splits=10)
# print(kf)
scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='accuracy')
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# # Predicting test Data

# In[15]:


# import numpy as nd

# Xnew = [[1,198,50]]
Xnew = sc.transform(X_test)
y_predict = clf.predict(X_test)
# for i in range(100):
#     print(str(y_predict [i]) + " " + str(y_test[i]))
for i in range(len(y_test)):
    y_test
y_predict = nd.array(y_predict ).tolist()
y_actual = []
for entry in y_test:
    y_actual.append(entry[0])
# print(type(y_predict))
# print(y_predict)
# print(type(y_actual))
# print(y_actual)


# In[16]:


naming = {'0': 'Extremely weak', '1': 'Weak', '2': 'Normal', '3': 'Overweight', '4': 'Obesity', '5': 'Extreme obesity'}
print(naming)
# print("Actual" , "Predicted")
# for i in range(100):
#     print(naming[y_actual[i]] + " , " + naming[y_predict[i]])


# # Writing the data to output file

# In[17]:


output_fields = ['Gender', 'Weight', 'Height', 'Index', 'Actual', 'Preidcted']
print (output_fields)
output_rows = []
for i in range(100):
    output_row = []
    output_row.extend(rows_X[i])
    output_row.extend(rows_y[i])
    output_row.append(naming[y_actual[i]])
    output_row.append(naming[y_predict[i]])
    output_rows.append(output_row)

# print(output_rows)

#name of csv file

file_name = "output.csv"

#writing to csv file

with open(file_name, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
      
    # writing the fields
    csvwriter.writerow(output_fields)
      
    # writing the data rows
    csvwriter.writerows(output_rows)


# # Analysis using confusion matrix

# In[18]:


# from sklearn import metrics
print(metrics.confusion_matrix(y_actual, y_test, labels=['0','1','2','3','4','5']))


# In[19]:


print(metrics.classification_report (y_actual, y_test, labels=['0','1','2','3','4','5']))


# In[ ]:





# In[ ]:





# In[ ]:




