# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:53:09 2019

@author: LABA
"""

from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import matplotlib.pyplot as plt

look = pd.read_csv('ratings.csv')

data = pd.read_csv('ratings.csv')

count = data['rating'].value_counts()

labels = pd.DataFrame({'rating':count.index, 'count':count.values})

x = labels['rating'].tolist()
y = labels['count'].tolist()

plt.bar(x, y, width=0.3)
plt.show()

labels2 = data.groupby('userId')['rating'].count()

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
train, test = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(train)
pred = model.test(test)
accuracy.rmse(pred, verbose=True)
