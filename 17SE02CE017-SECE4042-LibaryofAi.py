#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow.keras as keras
import tensorflow as tf

print(tf.__version__)


# In[4]:


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[5]:


print(x_train[0])


# In[6]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


# In[7]:


print(y_train[0])


# In[8]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[9]:


print(x_train[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


# In[10]:


model = tf.keras.models.Sequential()


# In[11]:


model.add(tf.keras.layers.Flatten())


# In[12]:


model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))


# In[13]:


model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# In[14]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[15]:


model.fit(x_train, y_train, epochs=3)


# In[16]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


# In[17]:


import tensorflow as tf


# In[18]:


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)  
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten())  
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)  
print(val_loss)  
print(val_acc)  


# In[19]:


model.save('epic_num_reader.model')


# In[20]:


new_model = tf.keras.models.load_model('epic_num_reader.model')


# In[21]:


predictions = new_model.predict(x_test)
print(predictions)


# In[22]:


import numpy as np

print(np.argmax(predictions[0]))


# In[23]:


plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()


# In[ ]:




