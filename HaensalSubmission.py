
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt


# In[2]:

dataframe = pd.read_csv('sample.csv',header=None)


# In[3]:

frameValues=dataframe.describe()


# In[4]:

X=dataframe.ix[:,0:294]
Y=dataframe.ix[:,295]


# Checking if any values are null

# In[5]:

dataframe.isnull().values.any()


# Checking if same value is present in all data points. We can delete these columns, since it doesnt not add any value for prediction model

# In[6]:


temp = np.transpose(frameValues.as_matrix())
itr=0
columnsWithSameValue =[]
for data in temp:
    if data[3] == data[7]:
        columnsWithSameValue.append(itr)
    itr+=1
    
    


# Build covariance matrix

# In[7]:

features = X.shape[1]
correfOutput= np.ndarray(shape=[features,features])
for i in range(features):
    for j in range(features):
        res=np.corrcoef(X[i],X[j])
        correfOutput[i][j]=res[0][1]
        correfOutput[j][i]=res[1][0]


# In[8]:

featuresCovariance={}
for i in range(features):
    for j in range(i+1,features):
        if abs(correfOutput[i][j])>0.7:
            if i in featuresCovariance:
                featuresCovariance[i].append(j)
            else:
                featuresCovariance[i]=[j]


# Following set of columns are highly correlated- Some times its better to remove highly correlated features.
# In order to remove the highly correlated features- have used Chi2 metrics to decide which column to remove

# In[9]:

featuresCovariance


# In[10]:

canBeRemoved = []
for key in featuresCovariance:
    feaImportanceScore = chi2(X.ix[:,key:key+1],Y)[0][0]
    toBeRemoved=key
    for fea2 in  featuresCovariance[key]:
        temp = chi2(X.ix[:,fea2:fea2+1],Y)[0][0]
        if temp < feaImportanceScore:
            feaImportanceScore= temp
            toBeRemoved=fea2
    canBeRemoved.append(toBeRemoved)
    


# In[11]:

#removing the columns/features
X.drop(columnsWithSameValue,inplace=True,axis=1)
X.drop(canBeRemoved,inplace=True,axis=1)


# Normalizing the features, using min max scaler and selecting top 100 features/columns

# In[13]:




# In[14]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
df_normalized = pd.DataFrame(np_scaled)
X=df_normalized


from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(f_classif,k=100)
X = kbest.fit_transform(X2,Y)
X=pd.DataFrame(X)


# Splitting the data into training and test/validation dataset 

# In[15]:

labels= pd.factorize(Y)[0]

splitRatio = int(len(X)*0.75)
train_features = X.ix[:splitRatio-1]
train_labels = labels[:splitRatio]
test_features = X.ix[splitRatio:]
test_labels = labels[splitRatio:]
print(len(X))
print(train_features.shape,train_labels.shape)
print(test_features.shape,test_labels.shape)


# Assign weights to loss function function - since we have imbalance data

# In[16]:

# train_features
import collections
val=collections.Counter(Y)
weights =[]
for e in val:
    weights.append(float(val[e])/len(Y))
weights


# Neural Network code 
# 
# Have used 2 layer perceptron : with first hidden units of 60 and second layer of 15
# Last layer using Softmax classifier
# Used weighted cross entropy loss as loss function
# Adam optimizer for optimizing the loss function
# 
# Used F1 score as performance measure - because of imbalance dataset
# 

# In[17]:

import tensorflow as tf


# In[18]:

inputData = tf.placeholder(tf.float32,[None,X.shape[1]])
target = tf.placeholder(tf.float32,[None,5])


# In[19]:

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def neuralNetwork(inputData,inputSize,outputSize,isSoftmax=False):
    
    weights = new_weights([inputSize,outputSize])
    biases = new_biases(outputSize)
    
    layer = tf.add(tf.matmul(inputData, weights), biases)
    layer = tf.nn.relu (layer)
    if isSoftmax:
        return tf.nn.softmax(layer)
    
    return layer


# In[20]:

hiddenUnit1=60
hiddenUnit2=15
classes=5
learning_rate = 0.00001
layer_1 = neuralNetwork(inputData,inputSize=train_features.shape[1],outputSize=hiddenUnit1)
layer_2 = neuralNetwork(layer_1,inputSize=hiddenUnit1,outputSize=hiddenUnit2)
outputLayer = neuralNetwork(layer_2,inputSize=hiddenUnit2,outputSize=classes,isSoftmax=True)


error = tf.nn.softmax_cross_entropy_with_logits(logits=outputLayer, labels=target)
# print(error)
# scaled_error = tf.matmul(error, [weights])
# cost = tf.reduce_mean(scaled_error)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputLayer, labels=target))
cost =tf.reduce_mean(-tf.reduce_sum(target*tf.log(outputLayer) + (1-target)*tf.log(1-outputLayer), reduction_indices=1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(outputLayer, 1), tf.argmax(target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[21]:

sess= tf.Session() 
init_op = tf.initialize_all_variables()
sess.run(init_op)

batchSize=10
display_step = 10000
step=0
epoch=0
maxEpoch = 10
while epoch < maxEpoch:
    itr=0;
    while itr < len(train_features):
        batch_Feature = train_features.ix[itr:itr+batchSize-1]
        batch_Labels = np.eye(classes)[train_labels[itr:itr+batchSize]]
        sess.run(optimizer, feed_dict={inputData: batch_Feature, target: batch_Labels})

        if itr % display_step == 0:
                loss, acc,err = sess.run([cost, accuracy,outputLayer], feed_dict={inputData: batch_Feature,
                                                                  target: batch_Labels})
                print ("Iter " + str(step*batchSize) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        itr+=batchSize
        step+=1
    batch_Labels = np.eye(classes)[train_labels]
    loss, acc = sess.run([cost, accuracy], feed_dict={inputData: train_features,
                                                                  target:batch_Labels })
    print ("Epoch  " + str(epoch) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    epoch+=1
#     break


# In[22]:

batch_Labels = np.eye(classes)[test_labels]
pred = sess.run(outputLayer,feed_dict={inputData: test_features,target:batch_Labels})


# In[23]:

predictedLabels=np.argmax(pred,axis=1)


# In[24]:

from sklearn.metrics import accuracy_score


# In[25]:

accuracy_score(test_labels,predictedLabels)


# In[32]:

from sklearn.metrics import f1_score
print(f1_score(test_labels,predictedLabels,average='micro'))
print(f1_score(test_labels,predictedLabels,average='macro'))


# In[ ]:




# In[ ]:



