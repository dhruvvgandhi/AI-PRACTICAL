#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import random


# In[29]:


#Tansition probabelity 
s_s = 0.8
s_r = 0.2
r_s = 0.4
r_r = 0.6

#prior probabelity
s = 0.67
r = 0.33

#Emission probabelity
s_h = 0.8
r_h = 0.2
s_g = 0.4
r_g = 0.6

mood = ['H','H','G','G','G','H']
Probabelity = []
whether = []

if mood[0] == 'H':
    Probabelity.append((s*s_h,r*r_h))
else:
    Probabelity.append((s*s_g,r*r_g))
    
for i in range(1,len(mood)):
    y_s,y_r = Probabelity[-1]
    if mood[i] == 'H':
        total_s = max(y_r*s_s*s_h,y_r*r_s*s_h)
        total_r = max(y_s*s_r*r_h,y_r*r_r*r_h)
        Probabelity.append((total_s,total_r))
    else:
        total_s = max(y_r*s_s*s_h,y_r*r_s*s_h)
        total_r = max(y_s*s_r*r_h,y_r*r_r*r_g)
        Probabelity.append((total_s,total_r))

for p in Probabelity:
    if p[0]>p[1]:
        whether.append('S')
    else:
        whether.append('R')

whether                        


# In[ ]:




