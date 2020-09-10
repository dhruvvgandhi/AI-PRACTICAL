#!/usr/bin/env python
# coding: utf-8

# In[43]:


a = input("Enter Your Name: ") # it takes user input


# In[44]:


def switch():
# This will guide the user to choose option
    print( " Hello " ,a ," Which libary You want to learn Press Acroding as below \n Press 1 for Tenserflow\n press 2 for Scikit-Learn \n press 3 for keras \n press 4 for Pandas \n press 5 for Numpy")
# This will take option from user    
    option = int(input("\n your option : "))

# If user enters invalid option then this method will be called 
    def default():
        print("\n Incorrect option")

# Dictionary Mapping
    dict = {
        1 : Tenserflow,
        2 : ScikitLearn,
        3 : keras,
        4 : pandas,
        5 : Numpy,
    }
    dict.get(option,default)() # get() method returns the function matching the argument

switch() 


# In[42]:


# Show the result of Tenserflow
def Tenserflow():
    def switch1():
    # This will guide the user to choose option
        print( " Hello " ,a," Which Part You want to learn in Tenserflow Press Acroding as below \n Press 1 for OverView Of Tenserflow\n press 2 for Functionalitiy Provideded by Tenserflow \n press 3 for MethodContains in Tenserflow \n press 4 for Basic Example of Tenserflow \n press 5 for Advantages of Tenserflow \n press 6 for Disadvantages of Tenserflow \n press 7 for Back")
    # This will take option from user    
        option = int(input("\n your option : "))

    # If user enters invalid option then this method will be called 
        def default():
            print("\n Incorrect option")

    # Dictionary Mapping
        dict = {
            1 : overviewTenserflow,
            2 : FunctionalitiyTenserflow,
            3 : MethodContainsTenserflow,
            4 : BasicExampleTenserflow,
            5 : AdvantagesOfTenserflow,
            6 : DisAdvantagesOfTenserflow,
            7 : switch,
        }
        dict.get(option,default)() # get() method returns the function matching the argument
    switch1() 


# In[1]:


def overviewTenserflow():
    print("\n Currently, the most famous deep learning library in the world is Google's TensorFlow. Google product uses machine learning in all of its products to improve the search engine, translation, image captioning or recommendations.")
    print("\nGoogle does not just have any data; they have the world's most massive computer, so Tensor Flow was built to scale. TensorFlow is a library developed by the Google Brain Team to accelerate machine learning and deep neural network research.")
    print("\n It was built to run on multiple CPUs or GPUs and even mobile operating systems, and it has several wrappers in several languages like Python, C++ or Java.")
    print("\n")
    Tenserflow()


# In[2]:


def FunctionalitiyTenserflow():
    print("\n tf.function ")
    print("\nTensor")
    print("\nTensorflow's name is directly derived from its core framework: Tensor. In Tensorflow, all the computations involve tensors. A tensor is a vector or matrix of n-dimensions that represents all types of data. All values in a tensor hold identical data type with a known (or partially known) shape. The shape of the data is the dimensionality of the matrix or array.")
    print("\nA tensor can be originated from the input data or the result of a computation. In TensorFlow, all the operations are conducted inside a graph. The graph is a set of computation that takes place successively. Each operation is called an op node and are connected to each other.")
    print("\nThe graph outlines the ops and connections between the nodes. However, it does not display the values. The edge of the nodes is the tensor, i.e., a way to populate the operation with data.")
    print("\nGraphs")
    print("\nTensorFlow makes use of a graph framework. The graph gathers and describes all the series computations done during the training. The graph has lots of advantages:")
    print("\nIt was done to run on multiple CPUs or GPUs and even mobile operating system")
    print("\nThe portability of the graph allows to preserve the computations for immediate or later use. The graph can be saved to be executed in the future.")
    print("\nAll the computations in the graph are done by connecting tensors together")
    print("\nA tensor has a node and an edge. The node carries the mathematical operation and produces an endpoints outputs. The edges the edges explain the input/output relationships between nodes.")
    print("\n")
    Tenserflow()


# In[3]:


def MethodContainsTenserflow():
    print("\nPopular Method uses For Algorithm")
    print("\nLinear regression: tf.estimator.LinearRegressor")
    print("\nClassification:tf.estimator.LinearClassifier")
    print("\nDeep learning classification: tf.estimator.DNNClassifier")
    print("\nDeep learning wipe and deep: tf.estimator.DNNLinearCombinedClassifier")
    print("\nBooster tree regression: tf.estimator.BoostedTreesRegressor")
    print("\nBoosted tree classification: tf.estimator.BoostedTreesClassifier")
    print("\n")
    Tenserflow()


# In[4]:


def BasicExampleTenserflow():
    print("\n import numpy as np")
    print("\nimport tensorflow as tf")
    print("\nDefine the variable")
    print("\nX_1 = tf.placeholder(tf.float32, name = X_1)")
    print("\nX_2 = tf.placeholder(tf.float32, name = X_2)")
    print("\nDefine the computation")
    print("\nmultiply = tf.multiply(X_1, X_2, name = multiply)")
    print("\nExecute the operation")
    print("\nwith tf.Session() as session:")
    print("\nresult = session.run(multiply, feed_dict={X_1:[1,2,3], X_2:[4,5,6]})")
    print("\nprint(result)")
    print("\nOUTPUT [ 4. 10. 18.]")
    print("\n")
    print("\nCreate the data")
    print("\nimport numpy as np")
    print("\nx_input = np.random.sample((1,2))")
    print("\nprint(x_input)")
    print("\n[[0.8835775 0.23766977]]")
    print("\nCreate the placeholder")
    print("\nx = tf.placeholder(tf.float32, shape=[1,2], name = 'X')")
    print("\nDefine the dataset method")
    print("\ndataset = tf.data.Dataset.from_tensor_slices(x)")
    print("\nCreate the pipeline")
    print("\niterator = dataset.make_initializable_iterator() get_next = iterator.get_next()")
    print("\nExecute the program")
    print("\nwith tf.Session() as sess:")  
    print("\nsess.run(iterator.initializer, feed_dict={ x: x_input })")  
    print("\nprint(sess.run(get_next))")
    print("\n[0.8835775  0.23766978]")
    print("\n")
    Tenserflow()


# In[5]:


def AdvantagesOfTenserflow():
    print("\nAdvantages of Tensorflow")
    print("\n1->Graphs")
    print("\nTensorflow has better computational graph visualizations, which are indigenous when compared to other libraries like Torch and Theano.")
    print("\n2->Library Management")
    print("\nBacked by Google, TensorFlow has the advantage of the seamless performance, quick updates and frequent new releases with new features.")
    print("\n3->Debugging")
    print("\nTensorflow lets you execute subparts of a graph which gives it an upper-hand as you can introduce and retrieve discrete data onto an edge and therefore offers great debugging method.")
    print("\n4->Scalability")
    print("\nThe libraries can be deployed on a gamut of hardware machines, starting from cellular devices to computers with complex setups.")
    print("\n5->Pipelining")
    print("\nTensorFLow is highly parallel and designed to use various backends software (GPU, ASIC) etc.")
    print("\n")
    Tenserflow()


# In[6]:


def DisAdvantagesOfTenserflow():
    print("\nDisadvantages of Tensorflow")
    print("\n1->Missing Symbolic Loops")
    print("\nThe feature that’s most required when it comes to variable length sequences are the symbolic loops. Unfortunately, TensorFlow does not offer this feature, but there is a workaround using finite unfolding (bucketing).")
    print("\n2->No support for Windows")
    print("\nThere is still a wide variety of users who are comfortable with a windows environment rather than a Linux in their systems and TensorFlow does not assuage these users. But, you need not worry if you are a Windows user as you can install it within a conda environment or using the python package library, pip.")
    print("\n3->Benchmark Tests")
    print("\nTensorFlow lacks behind in both speed and usage when compared to its competitors ")
    print("\n4-->No GPU support other than Nvidia and only language support")
    print("\nCurrently, the only supported GPUs are that of NVIDIA and the only full language support is of Python which makes it a downside as there is a rise of other languages in deep learning as well like Lau.")
    print("\n")
    Tenserflow()


# In[7]:


# Show the result of Scikit-Learn
def ScikitLearn():
    def switch2():
    # This will guide the user to choose option
        print( " Hello " ,a ," Which Part You want to learn in ScikitLearn Press Acroding as below \n Press 1 for OverView Of ScikitLearn\n press 2 for Functionalitiy Provideded by ScikitLearn \n press 3 for MethodContains in ScikitLearn\n press 4 for Basic Example of ScikitLearn\n press 5 for Advantages of ScikitLearn\n press 6 for Disadvantages of ScikitLearn\n press 7 for Back")
    # This will take option from user    
        option = int(input("\n your option : "))

    # If user enters invalid option then this method will be called 
        def default():
            print("\n Incorrect option")

    # Dictionary Mapping
        dict = {
            1 : overviewScikitLearn,
            2 : FunctionalitiyScikitLearn,
            3 : MethodContainsScikitLearn,
            4 : BasicExampleScikitLearn,
            5 : AdvantagesOfScikitLearn,
            6 : DisAdvantagesOfScikitLearn,
            7 : switch,
        }
        dict.get(option,default)() # get() method returns the function matching the argument
    switch2() 


# In[8]:


def overviewScikitLearn():
    print("\nScikit-learn is an open source Python library for machine learning. The library supports state-of-the-art algorithms such as KNN, XGBoost, random forest, SVM among others. It is built on top of Numpy. Scikit-learn is widely used in kaggle competition as well as prominent tech companies. Scikit-learn helps in preprocessing, dimensionality reduction(parameter selection), classification, regression, clustering, and model selection")
    print("\nScikit-learn is not very difficult to use and provides excellent results. However, scikit learn does not support parallel computations. It is possible to run a deep learning algorithm with it but is not an optimal solution, especially if you know how to use TensorFlow.")
    print("\n")
    ScikitLearn()    


# In[9]:


def FunctionalitiyScikitLearn():
    print("\nDimensionality Reduction")
    print("\nRegression")
    print("\nPreprocessing")
    print("\nClassification")
    print("\nModel Selection")
    print("\nClustering")
    print("\n")
    ScikitLearn()


# In[10]:


def MethodContainsScikitLearn():
    print("\n Not Found Yet, Coming soon !! ")
    print("\n")
    ScikitLearn()


# In[11]:


def BasicExampleScikitLearn():
    print("\nLoading an example dataset")
    print("\nfrom sklearn import datasets")
    print("\niris = datasets.load_iris()")
    print("\ndigits = datasets.load_digits()")
    print("\nprint(digits.data)")
    print("\n[[ 0.   0.   5. ...   0.   0.   0.]")
    print("\n[ 0.   0.   0. ...  10.   0.   0.]")
    print("\n[ 0.   0.   0. ...  16.   9.   0.]")
    print("\n...")
    print("\n[ 0.   0.   1. ...   6.   0.   0.]")
    print("\n[ 0.   0.   2. ...  12.   0.   0.]")
    print("\n [ 0.   0.  10. ...  12.   1.   0.]]")
    print("\ndigits.target")
    print("\narray([0, 1, 2, ..., 8, 9, 8])")
    print("\nLearning and predicting")
    print("\nfrom sklearn import svm")
    print("\nclf = svm.SVC(gamma=0.001, C=100.)")
    print("\nclf.fit(digits.data[:-1], digits.target[:-1])")
    print("\nSVC(C=100.0, gamma=0.001)")
    print("\nclf.predict(digits.data[-1:])")
    print("\narray([8])")
    print("\n")
    ScikitLearn()


# In[12]:


def AdvantagesOfScikitLearn():
    print("\nFree")
    print("\nSince scikit-learn is distributed under BSD license, it is free to use for anyone. In addition, with the minimal restrictions of its license, users wouldn’t have to worry about legal limitations when designing their platforms and applications.")
    print("\nEasy to use")
    print("\nMany research organizations and commercial industries have used scikit-learn in their operations and they all agree about how the module is easy to use. Because of that, they don’t run to any problem when performing a variety of processes.")3print("\nVersatile use")
    print("\nThe system is a handy tool that can do a multitude of things such as creating neuroimages, identifying abusive cloud actions, predicting consumer behavior, etc. It is widely used by research and commercial organizations throughout the world, proof of its versatility and ease of use.")
    print("\nBacked by international community")
    print("\nSince its one-man mission origin, scikit-learn has come a long way and is being developed by numerous authors in INRIA headed by Fabian Pedregosa as well as by various independent contributors. Because of this, the module is always updated, with several releases each year. scikit-learn is also backed by an international online community that users can count on if ever they run into troubles or have queries.")
    print("\nProperly documented")
    print("\nTo help ensure that new and old users both get the assistance they require with regards to integrating scikit-learn with their platforms, as well as extensive, detailed API documentation accessible from their website, is provided. With this, users can seamlessly integrate the learning algorithms in their own platforms.")
    print("\n")
    ScikitLearn()


# In[13]:


def DisAdvantagesOfScikitLearn():
    print("\n Not Found Yet, Coming soon !! ")
    print("\n")
    ScikitLearn()


# In[14]:


# Show the result of keras
def keras():
    def switch3():
    # This will guide the user to choose option
        print( " Hello " ,a ," Which Part You want to learn in keras Press Acroding as below \n Press 1 for OverView Of keras\n press 2 for Functionalitiy Provideded by keras \n press 3 for MethodContains in keras\n press 4 for Basic Example of keras\n press 5 for Advantages of keras\n press 6 for Disadvantages of keras\n press 7 for Back")
    # This will take option from user    
        option = int(input("\n your option : "))

    # If user enters invalid option then this method will be called 
        def default():
            print("\n Incorrect option")

    # Dictionary Mapping
        dict = {
            1 : overviewkeras,
            2 : Functionalitiykeras,
            3 : MethodContainskeras,
            4 : BasicExamplekeras,
            5 : AdvantagesOfkeras,
            6 : DisAdvantagesOfkeras,
            7 : switch,
        }
        dict.get(option,default)() # get() method returns the function matching the argument
    switch3() 


# In[15]:


def overviewkeras():
    print("\nKERAS is an Open Source Neural Network library written in Python that runs on top of Theano or Tensorflow. It is designed to be modular, fast and easy to use. It was developed by François Chollet, a Google engineer.")
    print("\nKeras doesn't handle low-level computation. Instead, it uses another library to do it, called the Backend. So Keras is high-level API wrapper for the low-level API, capable of running on top of TensorFlow, CNTK, or Theano.")
    print("\nKeras High-Level API handles the way we make models, defining layers, or set up multiple input-output models. In this level, Keras also compiles our model with loss and optimizer functions, training process with fit function. Keras doesn't handle Low-Level API such as making the computational graph, making tensors or other variables because it has been handled by the "backend" engine.")
    print("\n")
    keras()


# In[16]:


def Functionalitiykeras():
    print("\n Not Found Yet, Coming soon !! ")
    print("\n")
    keras()


# In[17]:


def MethodContainskeras():
    print("\nmodel.compile(optimizer='sgd', loss='mse', metrics=['mse'])")
    print("\n")
    keras()


# In[18]:


def BasicExamplekeras():
    print("\nKeras Fundamentals for Deep Learning")
    print("\nlinear regression")
    print("\nimport keras")
    print("\nfrom keras.models import Sequential")
    print("\nfrom keras.layers import Dense, Activation")
    print("\nimport numpy as np")
    print("\nimport matplotlib.pyplot as plt") 
    print("\nx = data = np.linspace(1,2,200)")
    print("\ny = x*4 + np.random.randn(*x.shape) * 0.3")
    print("\nmodel = Sequential()")
    print("\nmodel.add(Dense(1, input_dim=1, activation='linear'))")
    print("\nmodel.compile(optimizer='sgd', loss='mse', metrics=['mse'])")
    print("\nweights = model.layers[0].get_weights()")
    print("\nw_init = weights[0][0][0]")
    print("\nb_init = weights[1][0]")
    print("\nprint('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) ")
    print("\nmodel.fit(x,y, batch_size=1, epochs=30, shuffle=False)")
    print("\nweights = model.layers[0].get_weights()")
    print("\nw_final = weights[0][0][0]")
    print("\nb_final = weights[1][0]")
    print("\nprint('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))")
    print("\npredict = model.predict(data)")
    print("\nplt.plot(data, predict, 'b', data , y, 'k.')")
    print("\nplt.show()")
    print("\n")
    keras()


# In[45]:


def AdvantagesOfkeras():
    print("\n1 ->User-Friendly and Fast Deployment")
    print("\nKeras is a user-friendly API and it is very easy to create neural network models with Keras. It is good for implementing deep learning algorithms and natural language processing. We can build a neural network model in just a few lines of code.")
    print("\n2 ->Quality Documentation and Large Community Support")
    print("\nKeras has one of the best documentations ever. Documentation introduces you to each function in a very organized and sequential way. The codes and the examples given are very useful to understand the behavior of the method.")
    print("\nKeras also has great community support. There are lots of community codes on various open-source platforms. Many developers and Data Science enthusiasts prefer Keras for competing in Data Science challenges. So, we have a constant stream of insightful competition codes in Keras.Many of the researchers publish their codes and tutorials to the general public.")
    print("\n3 -> Multiple Backend and Modularity")
    print("\nKeras provides multiple backend support, where Tensorflow, Theano and CNTK being the most common backends. We can choose any of these backends according to the needs for different projects.")
    print("\nWe can also train the Keras model on one backend and test its results on other. It is very easy to change a backend in Keras, you just have to write the name of the backend in the configuration file.")
    print("\n4 ->Pretrained models")
    print("\nKeras provides some deep learning models with their pre-trained weights. We can use these models directly for making predictions or feature extraction.")   
    print("\nThese models have built-in weights, these weights are the results of training the model on ImageNet dataset.")
    print("\nSome of the available models are:")
    print("\nXception")
    print("\nVGG16")
    print("\nVGG19")
    print("\nResNet, ResNetV2")
    print("\nInceptionV3")
    print("\nInceptionResNetV2")
    print("\nMobileNet")
    print("\nMobileNetV2")
    print("\nDenseNet")
    print("\nNASNet")
    print("\n")
    print("\n5 -> Multiple GPU Support")
    print("\nKeras allows us to train our model on a single GPU or use multiple GPUs. It provides built-in support for data parallelism. It can process a very large amount of data.")
    print("\n")
    keras()


# In[20]:


def DisAdvantagesOfkeras():
    print("\n1. Problems in low-level API")
    print("\nSometimes you get low-level backend errors continuously and it becomes very irritating.  These errors occur because we may want to perform some operations that Keras was not designed for.
    print("\nIt does not allow to modify much about its backend. Error logs are difficult to debug.")
    print("\n2 -> Need improvement in some features")
    print("\nKeras data-preprocessing tools are not that much satisfying when we compare it with other packages like scikit-learn. It is not so good to build some basic machine learning algorithms like clustering and PCM (principal component analysis). It does not have features of dynamic chart creation.")
    print("\n3 -> Slower than its backend")
    print("\nSometimes it is slow on GPU and takes longer time in computation compared with its backends. So we may have to sacrifice speed for its user-friendliness.")    
    print("\n")
    keras()


# In[28]:


# Show the result of pandas
def pandas():
    def switch4():
    # This will guide the user to choose option
        print( " Hello " ,a ," Which Part You want to learn in pandas Press Acroding as below \n Press 1 for OverView Of pandas \n press 2 for Functionalitiy Provideded by pandas \n press 3 for MethodContains in pandas\n press 4 for Basic Example of pandas\n press 5 for Advantages of pandas\n press 6 for Disadvantages of pandas\n press 7 for Back")
    # This will take option from user    
        option = int(input("\n your option : "))

    # If user enters invalid option then this method will be called 
        def default():
            print("\n Incorrect option")

    # Dictionary Mapping
        dict = {
            1 : overviewpandas,
            2 : Functionalitiypandas,
            3 : MethodContainspandas,
            4 : BasicExamplepandas,
            5 : AdvantagesOfpandas,
            6 : DisAdvantagesOfpandas,
            7 : switch,
        }
        dict.get(option,default)() # get() method returns the function matching the argument
    switch4() 
        


# In[29]:


def overviewpandas():
    print("\nPandas is an opensource library that allows to you perform data manipulation in Python. Pandas library is built on top of Numpy, meaning Pandas needs Numpy to operate. Pandas provide an easy way to create, manipulate and wrangle the data. Pandas is also an elegant solution for time series data.")
    print("\n")
    pandas()


# In[30]:


def Functionalitiypandas():
    print("\nLoading And Saving Data")
    print("\nColumn Insertion And Deletion")
    print("\nData Selection")
    print("\nColumn And Row Renaming")
    print("\nRow Deletion")
    print("\nData Sorting")
    print("\nHandling Missing Value")
    print("\nHandling Duplicated Data")
    print("\nData Exploration")
    print("\nData Visualization") 
    print("\n")
    pandas()


# In[31]:


def MethodContainspandas():
    print("\n Not Found Yet, Coming soon !! ")
    print("\n")
    pandas()


# In[32]:


def BasicExample():
    print("\nCreating Series using list:")
    print("\nimport pandas as pd")
    print("\nser1 = pd.Series([1.5, 2.5, 3, 4.5, 5.0, 6])")
    print("\nprint(ser1)")
    print("\nCreating Series of string values with name:")
    print("\nimport pandas as pd") 
    print("\nser2 = pd.Series([India, Canada, Germany], name=Countries)")
    print("\nprint(ser2)")
    print("\nPython shorthand for list creation used to create Series:")
    print("\nimport pandas as pd")
    print("\nser3 = pd.Series([A]*4)")
    print("\nprint(ser3)")
    print("\nCreating Series using dictionary:")
    print("\nimport pandas as pd")
    print("\nser4 = pd.Series({India: New Delhi,Japan: Tokyo,UK:London})")
    print("\nprint(ser4)")
    print("\n")
    pandas()


# In[33]:


def AdvantagesOfpandas():
    print("\nAdvantages of Pandas Library")
    print("\nThere are many benefits of Python Pandas library, listing them all would probably take more time than what it takes to learn the library. Therefore, these are the core advantages of using the Pandas library:")
    print("\n1->Data representation")
    print("\nPandas provide extremely streamlined forms of data representation. This helps to analyze and understand data better. Simpler data representation facilitates better results for data science projects.")
    print("\n2->Less writing and more work done")
    print("\nIt is one of the best advantages of Pandas. What would have taken multiple lines in Python without any support libraries, can simply be achieved through 1-2 lines with the use of Pandas. Thus, using Pandas helps to shorten the procedure of handling data. With the time saved, we can focus more on data analysis algorithms.")
    print("\n3->An extensive set of features")
    print("\nPandas are really powerful. They provide you with a huge set of important commands and features which are used to easily analyze your data. We can use Pandas to perform various tasks like filtering your data according to certain conditions, or segmenting and segregating the data according to preference, etc.")
    print("\n4->Efficiently handles large data")
    print("\nWes McKinney, the creator of Pandas, made the python library to mainly handle large datasets efficiently. Pandas help to save a lot of time by importing large amounts of data very fast.")
    print("\n5->Makes data flexible and customizable")
    print("\nPandas provide a huge feature set to apply on the data you have so that you can customize, edit and pivot it according to your own will and desire. This helps to bring the most out of your data.")
    print("\n6->Made for Python")
    print("\nPython programming has become one of the most sought after programming languages in the world, with its extensive amount of features and the sheer amount of productivity it provides. Therefore, being able to code Pandas in Python, enables you to tap into the power of the various other features and libraries which will use with Python. Some of these libraries are NumPy, SciPy, MatPlotLib, etc.")
    print("\n")
    pandas()


# In[34]:


def DisAdvantagesOfpandas():
    print("\nDisadvantages of Pandas Library")
    print("\nEverything has its disadvantages as well, and it is important to know them, so, here are the disadvantages of using Pandas.")
    print("\n1 -> Steep learning curve")
    print("\nPandas initially have a mild learning slope. But as you go deeper into the library, the learning slope becomes steeper. The functionality becomes extremely confusing and can cause beginners some problems. However, with determination, it can be overcome.")
    print("\n2 -> Difficult syntax")
    print("\nWhile, being a part of Python, Pandas can become really tedious with respect to syntax. The code syntax of Pandas becomes really different when compared to the Python code, therefore people might have problems switching back and forth.")
    print("\n3 -> Poor compatibility for 3D matrices")
    print("\nIt is one of the biggest drawbacks of Pandas. If you plan to work with two dimensional or 2D matrices then Pandas are a Godsend. But once you go for a 3D matrix, Pandas will no longer be your go-to choice, and you will have to resort to NumPy or some other library.")
    print("\n4 -> Bad documentation")
    print("\nWithout good documentation, it becomes difficult to learn a new library. Pandas documentation isn’t much help to understand the harder functions of the library. Thus it slows down the learning procedure.")
    print("\n")
    pandas()


# In[35]:


# Show the result of Numpy
def Numpy():
    def switch5():
    # This will guide the user to choose option
        print( " Hello " ,a ," Which Part You want to learn in Numpy Press Acroding as below \n Press 1 for OverView Of Numpy\n press 2 for Functionalitiy Provideded by Numpy \n press 3 for MethodContains in Numpy\n press 4 for Basic Example of Numpy\n press 5 for Advantages of Numpy\n press 6 for Disadvantages of Numpy\n press 7 for Back")
    # This will take option from user    
        option = int(input("\n your option : "))

    # If user enters invalid option then this method will be called 
        def default():
            print("\n Incorrect option")

    # Dictionary Mapping
        dict = {
            1 : overviewNumpy,
            2 : FunctionalitiyNumpy,
            3 : MethodContainsNumpy,
            4 : BasicExampleNumpy,
            5 : AdvantagesOfNumpy,
            6 : DisAdvantagesOfNumpy,
            7 : switch,
        }
        dict.get(option,default)() # get() method returns the function matching the argument
    switch5() 


# In[36]:


def overviewNumpy():
    print("\nNumPy arrays are a bit like Python lists, but still very much different at the same time. For those of you who are new to the topic, let’s clarify what it exactly is and what it’s good for.")
    print("\nAs the name kind of gives away, a NumPy array is a central data structure of the numpy library. The library’s name is actually short for Numeric Python or Numerical Python.")
    print("\n")
    Numpy()


# In[37]:


def FunctionalitiyNumpy():
    print("\na powerful N-dimensional array object")
    print("\nsophisticated (broadcasting) functions")
    print("\ntools for integrating C/C++ and Fortran code")
    print("\nuseful linear algebra, Fourier transform, and random number capabilities")
    print("\n")
    Numpy()


# In[38]:


def MethodContainsNumpy():
    print("\nCharacteristics of NumPy")
    print("\nSupport n-dimensional arrays:")
    print("\n NumPy arrays are multidimensional or n-dimensional lists or matrices of fixed size with homogeneous elements( i.e. data type of all the elements in the array is the same).")
    print("\nRequire a contiguous allocation of memory:") 
    print("\nThis improves the efficiency as all elements of the array become directly accessible at a fixed offset from the starting of the array.")
    print("\nSupport selection using crop, slice, choose, etc.")
    print("\nSupport vectorized and complex operations.")
    print("\nSupport linear algebra, Fourier transform, and sophisticated broadcasting functions.")
    print("\n")
    Numpy()


# In[39]:


def BasicExample():
    print("\nimport numpy as np")
    print("\n#creating array using ndarray")
    print("\nA = np.ndarray(shape=(2,2), dtype=float)")
    print("\nprint('Array with random values:\n', A)")
    print("\n# Creating array from list")
    print("\nB = np.array([[1, 2, 3], [4, 5, 6]])")
    print("\nprint ('Array created with list:\n', B)")
    print("\n# Creating array from tuple")
    print("\nC = np.array((1 , 2, 3))")
    print("\nprint ('Array created with tuple:\n', C)")
    print("\n")
    Numpy()


# In[40]:


def AdvantagesOfNumpy():
    print("\nAdvantages of NumPy")
    print("\nThe core of Numpy is its arrays. One of the main advantages of using Numpy arrays is that they take less memory space and provide better runtime speed when compared with similar data structures in python(lists and tuples).")
    print("\nNumpy support some specific scientific functions such as linear algebra. They help us in solving linear equations.")
    print("\nNumpy support vectorized operations, like elementwise addition and multiplication, computing Kronecker product, etc. Python lists fail to support these features.")
    print("\nIt is a very good substitute for MATLAB, OCTAVE, etc as it provides similar functionalities and supports with faster development and less mental overhead(as python is easy to write and comprehend)")
    print("\nNumPy is very good for data analysis.")
    print("\n")
    Numpy()


# In[41]:


def DisAdvantagesOfNumpy():
    print("\nDisadvantages of NumPy")
    print("\nUsing 'nan' in Numpy: 'Nan' stands for 'not a number'.") 
    print("\nIt was designed to address the problem of missing values. NumPy itself supports “nan” but lack of cross-platform support within Python makes it difficult for the user. That’s why we may face problems when comparing values within the Python interpreter.")
    print("\nRequire a contiguous allocation of memory: ")
    print("\nInsertion and deletion operations become costly as data is stored in contiguous memory locations as shifting it requires shifting")
    print("\n")
    Numpy()

