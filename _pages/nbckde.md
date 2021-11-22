---
permalink: /projects/nbckde/
title: "Naive Bayes Classifier using KDE"
classes: wide
author_profile: true
---
```python
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
```

# The Data

To test the classifier, we will use a dataset that contains the season statistics of 275 NBA rookies between 2016 and 1980, and whether or not that player lasted five years or more in the league. The labeled data comes from (https://data.world/gmoney/nba-rookies-by-min-1980-2016). To add one more feature to the dataset (player efficiency rating), we import another dataset that contains this information (https://data.world/gmoney/nba-rookies-by-min-1980-2016).


```python
df = pd.read_csv('NBA.csv') # Import main dataset with labels
df = df.drop(index=29).reset_index(drop=True) # Charles Smith had repeat stats, drop one of his entries
df2 = pd.read_excel('NBA Rookies by Year.xlsx') # Import dataset with player efficiency column
df2 = df2[['Name','EFF','MIN']] # Remove all columns besides player name, efficiency, and minutes per game
df.rename(columns={"name": "Name","min":"MIN"},inplace=True) # Rename columns to use as join keys
df
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
      <th>X1</th>
      <th>Name</th>
      <th>gp</th>
      <th>MIN</th>
      <th>pts</th>
      <th>fgm</th>
      <th>fga</th>
      <th>fg</th>
      <th>3p_made</th>
      <th>3pa</th>
      <th>...</th>
      <th>ft</th>
      <th>oreb</th>
      <th>dreb</th>
      <th>reb</th>
      <th>ast</th>
      <th>stl</th>
      <th>blk</th>
      <th>tov</th>
      <th>target_5yrs</th>
      <th>probability_predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>Duane Cooper</td>
      <td>65</td>
      <td>9.9</td>
      <td>2.4</td>
      <td>1.0</td>
      <td>2.4</td>
      <td>39.2</td>
      <td>0.1</td>
      <td>0.5</td>
      <td>...</td>
      <td>71.4</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>2.3</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>1.1</td>
      <td>0</td>
      <td>0.525373</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>Dave Johnson</td>
      <td>42</td>
      <td>8.5</td>
      <td>3.7</td>
      <td>1.4</td>
      <td>3.5</td>
      <td>38.3</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>...</td>
      <td>67.8</td>
      <td>0.4</td>
      <td>0.7</td>
      <td>1.1</td>
      <td>0.3</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>0</td>
      <td>0.343556</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>Elmore Spencer</td>
      <td>44</td>
      <td>6.4</td>
      <td>2.4</td>
      <td>1.0</td>
      <td>1.9</td>
      <td>53.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>50.0</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.6</td>
      <td>1</td>
      <td>0.385416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>Larry Johnson</td>
      <td>82</td>
      <td>37.2</td>
      <td>19.2</td>
      <td>7.5</td>
      <td>15.3</td>
      <td>49.0</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>...</td>
      <td>82.9</td>
      <td>3.9</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>3.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.9</td>
      <td>0</td>
      <td>0.990362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>Mitch McGary</td>
      <td>32</td>
      <td>15.2</td>
      <td>6.3</td>
      <td>2.8</td>
      <td>5.2</td>
      <td>53.3</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>...</td>
      <td>62.5</td>
      <td>1.7</td>
      <td>3.5</td>
      <td>5.2</td>
      <td>0.4</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.439311</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>276</th>
      <td>1325</td>
      <td>Andre Spencer</td>
      <td>20</td>
      <td>21.1</td>
      <td>9.4</td>
      <td>3.7</td>
      <td>8.2</td>
      <td>44.8</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>...</td>
      <td>75.9</td>
      <td>1.9</td>
      <td>2.2</td>
      <td>4.1</td>
      <td>1.2</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>1.3</td>
      <td>0</td>
      <td>0.472822</td>
    </tr>
    <tr>
      <th>277</th>
      <td>1328</td>
      <td>Harold Miner</td>
      <td>73</td>
      <td>18.9</td>
      <td>10.3</td>
      <td>4.0</td>
      <td>8.4</td>
      <td>47.5</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>...</td>
      <td>76.2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>1.3</td>
      <td>0</td>
      <td>0.816702</td>
    </tr>
    <tr>
      <th>278</th>
      <td>1330</td>
      <td>Adam Keefe</td>
      <td>82</td>
      <td>18.9</td>
      <td>6.6</td>
      <td>2.3</td>
      <td>4.6</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>70.0</td>
      <td>2.1</td>
      <td>3.2</td>
      <td>5.3</td>
      <td>1.0</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>1.2</td>
      <td>1</td>
      <td>0.866027</td>
    </tr>
    <tr>
      <th>279</th>
      <td>1335</td>
      <td>Chris Smith</td>
      <td>80</td>
      <td>15.8</td>
      <td>4.3</td>
      <td>1.6</td>
      <td>3.6</td>
      <td>43.3</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>...</td>
      <td>79.2</td>
      <td>0.4</td>
      <td>0.8</td>
      <td>1.2</td>
      <td>2.5</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.8</td>
      <td>0</td>
      <td>0.730780</td>
    </tr>
    <tr>
      <th>280</th>
      <td>1336</td>
      <td>Brent Price</td>
      <td>68</td>
      <td>12.6</td>
      <td>3.9</td>
      <td>1.5</td>
      <td>4.1</td>
      <td>35.8</td>
      <td>0.1</td>
      <td>0.7</td>
      <td>...</td>
      <td>79.4</td>
      <td>0.4</td>
      <td>1.1</td>
      <td>1.5</td>
      <td>2.3</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>1.3</td>
      <td>1</td>
      <td>0.536788</td>
    </tr>
  </tbody>
</table>
<p>281 rows × 23 columns</p>
</div>



There are repeat names in this dataset such as Mark Davis, Charles Smith, Michael Anderson, and Michael Smith. To deal with these repeat names and ensure correct merging, we will join on two keys, name and minutes played per game


```python
df = df2.merge(df,how = 'inner',on=['Name','MIN'])
df = df.drop(columns = ['Name','X1','probability_predictions']) # Drop non-feature columns
```

Check for missing values in all columns.


```python
df.isnull().values.any()
```




    False



### Visualize the distribution of each feature

Even though Naive Bayes can be used with features that do not adhere to a Gaussian distribution, it is nonetheless interesting to see what distributions represent each feature.


```python
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.supylabel('Number of Rookies',fontsize=20)
for i in range(1, 21):
    plt.subplot(4, 5, i)
    column = df.columns[i-1]
    df[column] = pd.to_numeric(df[column])
    this = np.array(df[column])
    plt.hist(this,bins=19)
    plt.title(f"{column}",fontsize=16)
```


![png](/assets/images/output_10_0.png)


Split the data into training features, evaluation features, training labels, and evaluation labels.


```python
np_random = np.random.RandomState()
rand_unifs = np_random.uniform(0, 1, size = df.shape[0])
division_thresh = np.percentile(rand_unifs, 75)
train_indicator = rand_unifs < division_thresh
eval_indicator = rand_unifs >= division_thresh
```


```python
train_df = df[train_indicator].reset_index(drop=True)
train_features = np.array(train_df.loc[:,train_df.columns != 'target_5yrs'])
train_labels = np.array(train_df['target_5yrs'])
```


```python
eval_df = df[eval_indicator].reset_index(drop=True)
eval_features = np.array(eval_df.loc[:,eval_df.columns != 'target_5yrs'])
eval_labels = np.array(eval_df['target_5yrs'])
```

# Naive Bayes Classifier

The following is a from scratch implementation of a naive Bayes classifier.

Function to calculate the log of the prior probabilities. This will calculate the log of the prior probability of an NBA rookie for two conditions, either lasting at least five years in the NBA or exiting the league before their fifth season.


```python
def log_prior_probability(labels):
    size = labels.shape[0]
    positive_labels = labels.sum()
    negative_labels = size - positive_labels
    log_py = np.array([np.log(negative_labels/size),np.log(positive_labels/size)]).reshape(2,1)
    return log_py
```

Function to calculate the mean value for each feature. First column contains means for each feature for rookies that have lasted five years or longer in the NBA. Second column contains mean for each feature for rookies that have not lasted five years in the NBA.


```python
def calculate_mean(features, labels):
    featuresdf = pd.DataFrame(features)
    positive_feats = featuresdf[labels.astype(bool)].reset_index(drop=True)
    negative_feats = featuresdf[~labels.astype(bool)].reset_index(drop=True)
    avg_positive = np.array(positive_feats).mean(axis=0).reshape(-1,1)
    avg_negative = np.array(negative_feats).mean(axis=0).reshape(-1,1)
    return np.hstack((avg_negative,avg_positive))
```

Function to calculate the standard deviation for each feature under the two conditions.


```python
def calculate_std(features, labels):
    featuresdf = pd.DataFrame(features)
    positive_feats = featuresdf[labels.astype(bool)].reset_index(drop=True)
    negative_feats = featuresdf[~labels.astype(bool)].reset_index(drop=True)
    std_positive = np.std(np.array(positive_feats),axis=0).reshape(-1,1)
    std_negative = np.std(np.array(negative_feats),axis=0).reshape(-1,1)
    return np.hstack((std_negative,std_positive))
```

Function to calculate the log of the conditional probability. This will return the log of the calculated probability of an NBA rookie fitting into the given condition.


```python
def log_conditional_prob(features, avg, std, prior):
    log_arg = (1/(std*(np.sqrt(2*np.pi))))
    non_log_arg = -0.5*((features - avg)/std)**2
    one_feature = np.log(log_arg) + non_log_arg
    sum_probability = np.sum(one_feature, axis = 1)
    return sum_probability + prior
```

Function to wrap prior probabilities together in N x 2 array. The first column is the log of the probability that a player does not last five years in the NBA and the second column is the log of the probability that a player does last five years in the NBA. According to the classifier, the greater of these two numbers determines which condition the player is likely to fit into.


```python
def log_prob(train_features, mean_y, std_y, log_prior):
    negative_prob = log_conditional_prob(train_features, mean_y[:,0], std_y[:,0], log_prior[0,:][0]).reshape(-1,1)
    positive_prob = log_conditional_prob(train_features, mean_y[:,1], std_y[:,1], log_prior[1,:][0]).reshape(-1,1)
    return np.hstack((negative_prob,positive_prob))
```

Naive Bayes classifier class.


```python
class NBClassifier():
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.log_py = log_prior_probability(train_labels)
        self.mu_y = self.get_cc_means()
        self.sigma_y = self.get_cc_std()

    def get_cc_means(self):
        mu_y = calculate_mean(self.train_features, self.train_labels)
        return mu_y

    def get_cc_std(self):
        sigma_y = calculate_std(self.train_features, self.train_labels)
        return sigma_y

    def predict(self, features):
        log_p_x_y = log_prob(features, self.mu_y, self.sigma_y, self.log_py)
        return log_p_x_y.argmax(axis=1)
```

# Classifier Training and Evaluation

Train and evaluate NB classifier.


```python
nba_classifier = NBClassifier(train_features, train_labels)
train_pred = nba_classifier.predict(train_features)
eval_pred = nba_classifier.predict(eval_features)
```


```python
def readout(train_pred,eval_pred,train_labels,eval_labels):
    train_acc = (train_pred==train_labels).mean()
    eval_acc = (eval_pred==eval_labels).mean()
    print(f'Training data accuracy  : {train_acc:.10f}')
    print(f'Evaluation data accuracy: {eval_acc:.10f}')
```

Evaluating the classifier.


```python
readout(train_pred,eval_pred,train_labels,eval_labels)
```

    Training data accuracy  : 0.6165048544
    Evaluation data accuracy: 0.5942028986


Check to see if our implementation matches the naive Bayes classifier from the sklearn library.


```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(train_features, train_labels)
train_sk = gnb.predict(train_features)
eval_sk = gnb.predict(eval_features)
readout(train_sk,eval_sk,train_labels,eval_labels)
```

    Training data accuracy  : 0.6165048544
    Evaluation data accuracy: 0.5942028986


Good, the results from our implementation match the results from the stock algorithm from sklearn.

# Cross Validation

The above results represent the classifier accuracy for only one train-test split of the data. To better understand how the naive Bayes classifier performs with the data, we can use leave one out cross validation. This will show us the accuracy of the NB classifier when it acts on data that was not used to train it.


```python
def leave_one_out_cv(df,classifier, stock = False):
    df1 = np.array(df)
    loo_predictions = np.zeros(df1.shape[0])
    loo_train = np.zeros(df1.shape[0])
    for train_index, test_index in LeaveOneOut().split(df1):
        train_features_loo = df1[train_index]
        train_labels_loo = np.array(train_features_loo[:,-1])
        train_features_loo = train_features_loo[:,0:-1]
        eval_features_loo = df1[test_index]
        eval_labels_loo = eval_features_loo[:,-1]
        eval_features_loo = eval_features_loo[:,0:-1]
        if stock == False:
            _classifier = classifier(train_features_loo, train_labels_loo)
        else:
            _classifier = classifier
        train_pred = _classifier.predict(train_features_loo)
        eval_pred = _classifier.predict(eval_features_loo)
        train_acc = (train_pred==train_labels_loo).mean() # Calculate mean training accuracy
        eval_acc = (eval_pred==eval_labels_loo) # Calculate 1 (correct) or 0 (incorrect)

        loo_predictions[test_index[0]] = eval_acc #
        loo_train[test_index[0]] = train_acc

    print(f'Leave One Out Testing accuracy: {np.sum(loo_predictions)/df1.shape[0]:.10f}')
    print(f'Leave One Out Training accuracy: {np.sum(loo_train)/df1.shape[0]:.10f}')
```


```python
leave_one_out_cv(df,NBClassifier)
```

    Leave One Out Testing accuracy: 0.5963636364
    Leave One Out Training accuracy: 0.5973855342


# Naive Bayes with Kernel Density Estimation

A Gaussian naive Bayes classifier assumes that each feature in the dataset adheres to a normal distribution. From our visualization of the feature distributions, we can tell that this is not true of our dataset. Although naive Bayes classifiers can still be used in such situations, it stands to reason that if we used non-Gaussian probability density functions to model each feature's data, we could improve the performance of the classifier. To make such a classifier and build improved probability density functions, we will utilize kernel density estimation. For this, we will use the KernelDensity function from sklearn. Our probability density functions will now look as follows:


```python
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)
fig.supylabel('Probability',fontsize=20)

for i in range(1, 21):
    plt.subplot(4, 5, i)
    column = df.columns[i-1]
    x_d = np.linspace(-10, 110, 2000)
    x = np.array(df[column]).reshape(-1, 1)
    kde = KernelDensity(bandwidth=2.0, kernel='gaussian')
    kde.fit(x)
    logprob = kde.score_samples(x_d[:, None])
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
    plt.xlim((x.min()-4,x.max()+4))
    plt.ylim(-0.02, np.exp(logprob).max()+0.01)
    plt.title(f"{column}",fontsize=16)
    plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=0.1)
```


![png](/assets/images/output_44_0.png)


To get the best performance out of the improved classifier, we need to tune the bandwidth hyperparameter for the Kernel Density Estimation with our specific dataset. We can do this using the GridSearchCV function from sklearn.


```python
from sklearn.model_selection import GridSearchCV
bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths})
grid.fit(df)
grid.best_params_
```




    {'bandwidth': 2.25701971963392}



Rather than calculating the mean and standard deviation of each feature (under each of the two conditions) to define the Gaussian distributions, we instead pass the features (under each of the two conditions) to the kernel density estimation function in order to define a probability density function that more accurately represents the data.


```python
def create_models(training_fts,train_labels):
    featuresdf = pd.DataFrame(training_fts)
    pos_fts = np.array(featuresdf[train_labels.astype(bool)].reset_index(drop=True))
    neg_fts = np.array(featuresdf[~train_labels.astype(bool)].reset_index(drop=True))
    pos_models = np.array([KernelDensity(bandwidth = 2.25701971963392).fit(pos_fts[:,i].reshape(-1,1)) for i in range(training_fts.shape[1])])
    neg_models = np.array([KernelDensity(bandwidth = 2.25701971963392).fit(neg_fts[:,i].reshape(-1,1)) for i in range(training_fts.shape[1])])
    return np.hstack((neg_models.reshape(-1,1),pos_models.reshape(-1,1)))

def sum_log_likelihood(features, models, prior):
    logprobs = np.zeros(features.shape)
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            logprobs[i,j] = models[j].score(features[i,j].reshape(-1,1))
    return np.sum(logprobs,axis=1) + prior

def log_probability(features,models,log_py):
    neg_prediction = sum_log_likelihood(features,models[:,0], log_py[0,:][0]).reshape(-1,1)
    pos_prediction = sum_log_likelihood(features,models[:,1], log_py[1,:][0]).reshape(-1,1)
    return np.hstack((neg_prediction,pos_prediction))
```

Class for naive Bayes classifier with kernel density estimation.


```python
class KDEClassifier():
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.log_py = log_prior_probability(train_labels)
        self.models = create_models(self.train_features, self.train_labels)

    def predict(self, features):
        predictions = log_probability(features, self.models, self.log_py)
        return predictions.argmax(axis=1)
```

# KDE Classifier Training and Evaluation


```python
kde_classifier = KDEClassifier(train_features, train_labels)
train_pred = kde_classifier.predict(train_features)
eval_pred = kde_classifier.predict(eval_features)
```


```python
readout(train_pred,eval_pred,train_labels,eval_labels)
```

    Training data accuracy  : 0.7184466019
    Evaluation data accuracy: 0.5942028986


# Cross Validation of KDE classifier


```python
leave_one_out_cv(df,KDEClassifier)
```

    Leave One Out Testing accuracy: 0.6654545455
    Leave One Out Training accuracy: 0.6900597213


Compared to the stock naive Bayes classifier, our improved KDE naive Bayes classifier performed about 6% better. Not incredible, but a notable improvement over the base model. To see how other classifiers perform on this dataset, we can use several prebuilt classifiers from the sklearn library. We will also perform leave one out cross validation to preclude any aberrant results.

# Comparison to other stock sklearn classifiers


```python
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(train_features, train_labels)
leave_one_out_cv(df,dtc,stock = True)
```

    Leave One Out Testing accuracy: 0.9090909091
    Leave One Out Training accuracy: 0.9090909091



```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300).fit(train_features, train_labels)
leave_one_out_cv(df,rfc,stock = True)
```

    Leave One Out Testing accuracy: 0.9163636364
    Leave One Out Training accuracy: 0.9163636364



```python
from sklearn.neighbors import KNeighborsClassifier
knnc = KNeighborsClassifier(n_neighbors=1).fit(train_features, train_labels)
leave_one_out_cv(df,knnc,stock = True)
```

    Leave One Out Testing accuracy: 0.9018181818
    Leave One Out Training accuracy: 0.9018181818



```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'none',max_iter=10000).fit(train_features, train_labels)
leave_one_out_cv(df,lr,stock = True)
```

    Leave One Out Testing accuracy: 0.7381818182
    Leave One Out Training accuracy: 0.7381818182


Clearly, the naive Bayes classifier and the KDE naive Bayes classifier are not the best options when it comes to binary classification. Nonetheless, this dataset does not play to the strongsuits of naive Bayes classifiers. For example, the features in our dataset are not independent. That is, usually a stellar player will have stellar stats in most categories. Naive Bayes can actually perform admirably in situations where features are indeed independent (not just assumed to be independent) and may also perform well with categorical features rather than continuous features.
