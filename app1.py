import pandas as pd,numpy as np
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('new_data_stroke_n.csv')
df.head()
target = 'stroke'
encode = ['hypertension', 'heart_disease','gender', 'ever_married','work_type', 'Residence_type', 'smoking_status']

for col in encode:
    dummy = pd.get_dummies(df[col],prefix = col)
    df = pd.concat([df,dummy],axis = 1)
    del df[col]
target_mapper = {'positive_stroke': 1,
                 'negative_stroke': 0}
def target_encode(val):
    return target_mapper[val]

df['stroke'] = df['stroke'].apply(target_encode)

#df.age.astype('float32')
#df.bmi.astype('float32')
#df.avg_glucose_level.astype('float32')
X = df.drop('stroke', axis = 1)
y = df.stroke
clf = RandomForestClassifier()
clf.fit(X,y)

import pickle
pickle.dump(clf,open('stroke_clf_newest.pkl','wb'))