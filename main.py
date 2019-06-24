import functions as func
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

# load OnlineNewsPopularity file
unsampled_news_pop = pd.read_csv('OnlineNewsPopularityProcessed.csv')
# print(unsampled_news_pop)

# strip out space before column names
unsampled_news_pop.columns = [i.strip() for i in unsampled_news_pop.columns]
# print(list(unsampled_news_pop.columns))

"""
Data Visualisation
"""
# plot histogram
bin_num = int(len(unsampled_news_pop)/100)
plt.hist(unsampled_news_pop['shares'], bins=bin_num, log=True)
plt.title("Histogram")
plt.ylabel('Frequency')
plt.xlabel('Shares')
plt.tight_layout()
plt.show()

# find mean number of shares
shares_mean = unsampled_news_pop.loc[:,'shares'].mean() # 3395
# print(shares_mean)

# find top 1% of number of shares - these articles are 'popular'
one_pct = int(len(unsampled_news_pop)*0.01)
top_one_pct = unsampled_news_pop.nlargest(one_pct, 'shares')
# print(top_one_pct['shares'])
popular_thresh = top_one_pct['shares'].min() # 31900

# find top 0.1% of number of shares - these articles are 'viral'
pt_one_pct = int(len(unsampled_news_pop)*0.001)
top_pt_one_pct = unsampled_news_pop.nlargest(pt_one_pct, 'shares')
# print(top_pt_one_pct['shares'])
viral_thresh = top_pt_one_pct['shares'].min() # 115700

# visualise data
news_length = list(range(0, len(unsampled_news_pop)))
plt.scatter(news_length, unsampled_news_pop['shares'], c='k', marker='.')

# plot threshold lines
plt.axhline(y=shares_mean, color='r', linestyle='-', label='mean')
plt.axhline(y=popular_thresh, color='b', linestyle='-', label='popular')
plt.axhline(y=viral_thresh, color='g', linestyle='-', label='viral')

# plot labels
plt.title("Visualisation")
plt.xlabel('Article Number')
plt.ylabel('Number of shares')
plt.legend()
plt.tight_layout()
plt.show()

"""
Data Preparation
"""
# check data types to determine which columns to remove immediately
# print(unsampled_news_pop.dtypes)

# binarise 'shares' in new column 'shares_bin' (around 'popular' threshold)
unsampled_news_pop['shares_bin'] = (unsampled_news_pop['shares'] > popular_thresh).astype(int)
# print(news_pop['shares_bin'])

# check to make sure data has binarised correctly
bin_test = int(len(unsampled_news_pop)*0.002)
bin_testing = unsampled_news_pop.nlargest(bin_test, 'shares')
# print(bin_testing)

# undersample majority class
total = len(unsampled_news_pop)
nb_popular = unsampled_news_pop['shares_bin'].sum()
nb_non = total - nb_popular
news_pop_popular = unsampled_news_pop.loc[unsampled_news_pop['shares_bin'] == 1]
news_pop_non = unsampled_news_pop.loc[unsampled_news_pop['shares_bin'] == 0].sample(nb_popular)
news_pop = pd.concat((news_pop_popular, news_pop_non))
# these numbers should be the same
print('The sizes of the minority (popular) class and majority (non-viral) class are {}'.format((len(news_pop_popular), len(news_pop_non))))

# drop object column (url) and python timing column (timedelta)
news_pop = news_pop.drop(['url', 'timedelta'], axis=1)

# now remove 'shares' as this is what we test for
news_pop = news_pop.drop(['shares'], axis=1)
# print(news_pop)

# split into train,validation and test sets
train, other = train_test_split(news_pop, test_size=0.2, random_state=0)
validation, test = train_test_split(other, test_size=0.5, random_state=0)
print('The sizes for train, validation and test should be {}'.format((len(train), len(validation), len(test))))

"""
Model Selection
"""
class Finished(Exception):pass

# split sets into x and y folds
X_train = train.drop(columns=['shares_bin'])
y_train = train['shares_bin']

X_val = validation.drop(columns=['shares_bin'])
y_val = validation['shares_bin']

X_test = test.drop(columns=['shares_bin'])
y_test = test['shares_bin']

# # automatic forward selection
# columns_to_test = X_train.columns
# columns_in_model = []
# for i in range(0,60):
#     columns_in_model_updated, acc_best = func.select_column_to_add(X_train, y_train, X_val, y_val, columns_in_model, columns_to_test)
#     columns_in_model = columns_in_model_updated
#
# # automatic backward selection
# columns_to_test = X_train.columns
# columns_in_model = columns_to_test
# for i in range(0,60):
#     columns_in_model_updated, acc_best, columns_in_model = func.select_column_to_remove(X_train, y_train, X_val, y_val, columns_in_model, columns_to_test)
#     columns_to_test = columns_in_model_updated
#     columns_in_model = columns_in_model_updated

"""
Decision Tree
"""
dt = DecisionTreeClassifier(max_depth = 3)
dt = dt.fit(X_train, y_train)

dot_data = tree.export_graphviz(dt, out_file=None)
graph = graphviz.Source(dot_data)

predictors = X_train.columns
#print(predictors)
dot_data = tree.export_graphviz(dt, out_file=None,
                                feature_names = predictors,
                                class_names = ('Negative', 'Positive'),
                                filled = True, rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph

"""
Model testing
"""
selected_attributes = ['kw_max_avg', 'data_channel_is_world', 'data_channel_is_entertainment', 'LDA_03']
X = news_pop[selected_attributes]
y = news_pop['shares_bin'].values.reshape(-1,1)

mylr = logreg()
mylr.fit(X, y)

model_summary = func.ModelSummary(mylr, X, y)
model_summary.get_summary()
