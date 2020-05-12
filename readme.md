---
title: Term Project
date: 2020-05-11

# Put any other Academic metadata here...
---


# Term Project


Lijun Wang      
1001778502

The dataset we used is the board-game-geek-reviews data. Project's goal is given the review, predict the rating. 

### loading and preprocessing data 
Because our goal is given the review, predict the rating. We only need to use the bgg-13m-reviews.csv. 


```python
import numpy as np 
import pandas as pd 

import os
data = pd.read_csv('/kaggle/input/boardgamegeek-reviews/bgg-13m-reviews.csv')
len(data)
```




    13170073



we can see there are 13170073 record in this table.


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>user</th>
      <th>rating</th>
      <th>comment</th>
      <th>ID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>sidehacker</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Varthlokkur</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>dougthonus</td>
      <td>10.0</td>
      <td>Currently, this sits on my list as my favorite...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>cypar7</td>
      <td>10.0</td>
      <td>I know it says how many plays, but many, many ...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>ssmooth</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
  </tbody>
</table>
</div>



I show the first five record in this table. We can see there are some NaN value in comment column. Because we want to get rating based comment. If the comment is NaN, this record is useless. We need remove it.


```python
data1 = data.dropna(axis=0,how='any')
```

There are 2637756 records left.


```python
len(data1)
```




    2637756



I show the first-five records after drop those records that commends equals NaN. 


```python
data1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>user</th>
      <th>rating</th>
      <th>comment</th>
      <th>ID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>dougthonus</td>
      <td>10.0</td>
      <td>Currently, this sits on my list as my favorite...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>cypar7</td>
      <td>10.0</td>
      <td>I know it says how many plays, but many, many ...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>hreimer</td>
      <td>10.0</td>
      <td>i will never tire of this game.. Awesome</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>daredevil</td>
      <td>10.0</td>
      <td>This is probably the best game I ever played. ...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>hurkle</td>
      <td>10.0</td>
      <td>Fantastic game. Got me hooked on games all ove...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
  </tbody>
</table>
</div>



Show the description of those records.


```python
data1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>rating</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.637756e+06</td>
      <td>2.637756e+06</td>
      <td>2.637756e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.165498e+06</td>
      <td>6.852070e+00</td>
      <td>6.693990e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.926932e+06</td>
      <td>1.775769e+00</td>
      <td>7.304447e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000e+00</td>
      <td>1.401300e-45</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.691259e+06</td>
      <td>6.000000e+00</td>
      <td>3.955000e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.251858e+06</td>
      <td>7.000000e+00</td>
      <td>3.126000e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.737738e+06</td>
      <td>8.000000e+00</td>
      <td>1.296220e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.871082e+06</td>
      <td>1.000000e+01</td>
      <td>2.724090e+05</td>
    </tr>
  </tbody>
</table>
</div>



We can see the value of rating is between 1 and 10, and there are many decimal fraction. And the predicted value is also between 1 and 10, so this is a regression task. And because the number of records is very large. I randomly selected 1000 records from the data set as experimental data.


```python
data2 = data1.sample(1000)
```

Visual distrubution of ratings


```python
import matplotlib.pyplot as plt

n, bins, patches = plt.hist(data2.rating, 100)

plt.title('Distrubution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```

![png](https://raw.githubusercontent.com/lxw8502/academic-kickstart/master/content/post/Term%20Project_18_0.png)


Remove punctuation and special characters from comments. The word_list filled with the split comments.


```python
import re
word_list = []
for text in data2['comment']:
    text = re.sub('\n', '', text)  
    text = re.sub('[\s+\.\!\/_,$%;^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+', ' ', text)
    word_list.append(' '.join(list(text.strip().split())))
print(word_list[:10])
```

    ['A cousin to Through the desert not as deep but nice Very short rules and fast game', 'Only played a short demo during a game fair Waiting to play a full game to adjust my vote I like the theme', 'Great little card game', 'Kickstarter - ETA Augustus 2017: this became January 2018', 'The game has a lot going for it but the high luck factor really disappointed me', 'Love the concept of the game but not the rules', 'After a few games I have strong desire to play more to figure out the best strategy', '', 'TTR minus route building plus a bit of pick-up-and-deliver Bonus points for stylish presentation and quick playtime but luck of the draw seemed too decisive and the mechanical innovations didn t justify the complexity bump Entirely possible that there s subtleties that whooshed over my head in a first play seeing as I got trounced) that might elevate this from a well-produced pleasant pastime to a game worthy of attention Ultimately games of economics should have more economics if you ask me -- Cinque Terre with its fixed albeit randomized) prices just doesn t feel dynamic enough to me', 'A real surprise After reading the rules I thought for sure that this was going to be a mindnumbing repeatitive family game it s much deeper - requires much more thinking tactics than Ticket to Ride Quite clever unique actually Families will learn to play this game quickly but it will be the real gamers who will win this Be warned: Those who suffer from analysis paralysis should stay clear of this one because optimizing your move is the key to success Game is a bit repeatitive though towards the end and quite fiddly throughout Luck is relatively minimal Like someone said this is NOT a majority game Probably not good with 5-players too much downtime and too little control - great with 3 Rating goes up after a couple plays My early pick for SdJ along with Niagara']
    

Use sklearn train_test_split method to divide it into training set and test set.


```python
from sklearn.model_selection import train_test_split

target = data2['rating']
x_train,x_test,y_train,y_test=train_test_split(word_list,target,test_size=0.2,random_state=24)

print(x_train[:10])

```

    ['After several plays with or without the traitor this game is hard I think that being a little too eager to finish quests before they are necessary did us in as completing them doesn t stop the onslaught of black cards And if you re reading this I hate you Morgan My first play was the only one to use the traitor We had it rough trying to manage all of the possible quests and it turned out to be a good test of teamwork In the end we suspected a traitor but couldn t afford to be wrong in accusing him We should ve done it anyway', 'There is some game here but not a whole lot All in all I found this to be not bad but certainly not to the lofty levels that The Cauldron is at', 'My first almost card driven game And I still enjoy it', '3 a 5 jugadores - 1 hora', 'No Box', 'Review: https: boardgamegeek com thread 1168221 6-out-10-simple-game-zombie-slaughter-it-too-simpl', 'Not as much terrain as the Rise of the Valkyrie master set and the swamp tiles are not as useful as water tiles Lots of common squad marro figures especially the cool marro dog-like creatures', 'Vin d jeu: french review of That s a question: http: www vindjeu eu 2017 12 12 thats-a-question', 'Not bad Even fun for a while But ultimately I d rather play Settlers if I m in the mood for a dry Euro', 'Novel board and waterfall mechanic but all games played end ina procession - possibly fun for children']
    

List stop words to be removed


```python
stopwords = ['',',','"','.','+','-','!','?','*','/','@','1','2','3','4','5','6','7','8','9','#','$','%','&','^','(',')',':','...','a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',  'cannot', 'could', 'did', 'do', 'does', 'doing', 'down', 'during',
 'each', 'few', 'for', 'from', 'further', 
 'had', 'has', 'have', 'having', 'he', 'her', 'here',
 'hers', 'herself', 'him', 'himself', 'his', 'how',
 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
 'me', 'more', 'most', 'my', 'myself',
 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', 'she', 'should', 'so', 'some', 'such', 
 'than', 'that', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 
 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'was', 'we', 'were', 'what', 'when', 'where',
 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would',
 'you', 'your', 'yours', 'yourself', 'yourselves', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand']
```

Use CountVectorizer and TfidfTransformer to vectorize those comments.     
The model I used is RandomForestRegressor. A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.


### Fitting the model


```python
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

reg = Pipeline([('vect',CountVectorizer(stop_words=stopwords, ngram_range=(1,3), token_pattern=u'[a-zA-Z]+', analyzer = 'word')),
                  ('tfidf',TfidfTransformer(use_idf=True,norm='l2')),
                  ('dense',FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                  ('reg',RandomForestRegressor(n_estimators=20,max_depth=10,min_samples_split=8,max_features=None))])

scores = cross_val_score(reg, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

```

The actual mean squared error is simply the positive version of the number you're getting. The unified scoring API always maximizes the score, so scores which need to be minimized are negated in order for the unified scoring API to work correctly. The score that is returned is therefore negated when it is a score that should be minimized and left positive if it is a score that should be maximized.


```python
print("neg_mean_squared_error: %0.2f (+/- %0.2f) "   
        % (scores.mean(), scores.std()))
```

    neg_mean_squared_error: -3.13 (+/- 0.72) 
    


```python
scores
```




    array([-2.26485484, -2.57459162, -2.99601265, -3.49517072, -4.30621316])



### Export the model


```python
from sklearn.externals import joblib

joblib.dump(reg,'E:/RandomForest_reg.pkl')
```

### Reference

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor       

https://stackoverflow.com/questions/21443865/scikit-learn-cross-validation-negative-values-with-mean-squared-error      

