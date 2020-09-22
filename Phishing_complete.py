import pandas as pd
path = 'C:\\Users\\aradh\\Downloads\\Phishing_new.csv'
df = pd.read_csv(path)

print(df)

print(df.shape)

c = df.columns
print(c)

print(df.head(10))

print(df.sample(10))

print(df.sample(10).T)

print(df.describe().T)

#for a in c:
 #print(df.hist(column= a))
 #print(df.boxplot(column= a))

print(df.isnull().sum())

for b in c:
 value =df[b].unique()
 print(value)

x=df.loc[:, 'having_IP_Address':'Statistical_report' ]
y=df.loc[:, 'Result']
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
lo = logreg.fit(xTrain,yTrain)
print(lo.coef_)

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(xTrain,yTrain)
ypred=logreg.predict(xTest)


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(yTest, ypred)
element = cnf_matrix[0][0]
element_2 = cnf_matrix[0][1]

from sklearn.metrics import mean_squared_error
print(mean_squared_error(yTest, ypred))

from sklearn import metrics
print(metrics.classification_report(yTest, ypred))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
print(plt.title('Confusion matrix', y=1.1))
print(plt.ylabel('Actual label'))
print(plt.xlabel('Predicted label'))
plt.show()

accuracy = metrics.accuracy_score(yTest, ypred)
print(accuracy)

Precision = metrics.precision_score(yTest, ypred)
print(Precision)

Recall_phishing = metrics.recall_score(yTest, ypred)
print(Recall_phishing)

recall_non_phishing = element / (element + element_2)

m = 0.5 *((1/recall_non_phishing) + (1/Recall_phishing))
average_class_accuracy_hm = 1/m
print(average_class_accuracy_hm)

f1 = 2* Precision * Recall_phishing
f = Precision + Recall_phishing
f1_measure = f1 / f
print(f1_measure)

import matplotlib.pyplot as plt1
y_pred_proba = logreg.predict_proba(xTest)[::,1]
fpr, tpr, _ = metrics.roc_curve(yTest,  y_pred_proba)
auc = metrics.roc_auc_score(yTest, y_pred_proba)
plt1.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt1.legend(loc=4)
plt1.show()







