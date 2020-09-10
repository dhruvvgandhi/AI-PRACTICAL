#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = input("Enter Your Name: ")


# In[ ]:


def switch():
# This will guide the user to choose option
    print( " Hello " ,a ,"Can please tell us cancer report is positive or negative \n Press 1 for Positive \n Press 2 For Negative ")
# This will take option from user    
    option = int(input("\n your option : "))

# If user enters invalid option then this method will be called 
    def default():
        print("\n Incorrect option")

# Dictionary Mapping
    dict = {
        1 : positive,
        2 : nagative,
    }
    dict.get(option,default)() # get() method returns the function matching the argument

switch() 


# In[3]:


def positive():
    c = int(input("As We see " + a + " Your Test Report is Positive , Can Please provide test report percentage ? "))
    print()
    d = float(input( " " + a + " Can please tell us total number of population who having Cancer in percentage ? "))
    print()
    e = (c/100)
    f = (1-e)
    g = (d/100)
    cancer = ((e*g)/((e*g)+(f*e)))
    z=print("As We see " + a + " based on your input we can say that you have chance to have cancer is")
    print()
    print(cancer)
    print()
    print("Do You Want to try again")
    print()
    switch()


# In[4]:


def nagative():
    c = int(input("As We see " + a + " Your Test Report is Nagative,  Can Please provide test report percentage ? "))
    print()
    d = float(input( " " + a + " Can please tell us total number of population who having Cancer in percentage ? "))
    print()
    e = (c/100)
    f = (1-e)
    g = (d/100)
    cancer = ((e*g)/((e*g)+(f*e)))
    z=print("As We see " + a + " based on your input we can say that you have chance to have cancer is")
    print()
    print(cancer)
    print()
    print("Do You Want to try again")
    print()
    switch()


# In[ ]:




