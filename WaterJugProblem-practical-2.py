#!/usr/bin/env python
# coding: utf-8

# In[19]:


import networkx as nx #to generate graph
import queue as q #create a path to add accpt middleware or suceessser 
import matplotlib.pyplot as plt

def generateTree(jug1,jug2):
    class rules:
        def __init__(self,j1,j2): #j1 for jug->1 and #j2 for jug->2
            self.j1=j1 #pass the value of j1 enter by user to code
            self.j2=j2 #pass the value of j2 enter by user to code
            
         #operation of jug 1
        #Empty Opreation
        def empty_j1(self,state): #j1 is Empty means for jug->1  is empty
            state=list(state)
            state[0]=0 #jug->1 is empty means x=0
            return tuple(state)
        #Fill opertation on jug1
        def fill_j1(self,state):
            state=list(state)
            state[0]=self.j1 #jug->1 is fill means x=jug1
            return tuple(state) 
        #operation of jug 2
        #Empty Opreation 
        def empty_j2(self,state):
            state=list(state)
            state[1]=0#jug->2 is empty means y=0
            return tuple(state)
        #Fill opertation on jug2
        def fill_j2(self,state):
            state=list(state)
            state[1]=self.j2 #jug->2 is fill means y=jug2
            return tuple(state)
        #Tranfer jug1 to jug2
        #x+y <=jug2 =====> x=0 & y=x+y  
        #x+y >jug2  =====> [x=x-(y-3)] and [y=3]
        def trans_j1toj2(self,state):
            state=list(state)
            state[0],state[1]=max(0,state[1]+state[0]-self.j2),min(self.j2,state[1]+state[0])
            return tuple(state) 
        #Tranfer jug2 to jug1
        #x+y <=hug1 =====> x=x+y & y=0 
        #x+y >hug1  =====> [X=4] and [y=y-(4-x)]
        def trans_j2toj1(self,state):
            state=list(state)
            state[0],state[1]=min(self.j1,state[1]+state[0]),max(0,state[1]+state[0]-self.j1)
            return tuple(state)
        #combine all possibility 
        def applyAll(self,state):
            l=[]
            l.append(self.empty_j1(state))
            l.append(self.fill_j1(state))
            l.append(self.empty_j2(state))
            l.append(self.fill_j2(state))
            l.append(self.trans_j1toj2(state))
            l.append(self.trans_j2toj1(state))
            return l
    rule=rules(jug1,jug2)            
    node=(0,0)
    G=nx.Graph()
    G.add_node(node)
    Q=q.Queue(50)
    while(True):
        l=rule.applyAll(node)
        for i in l:
            if not G.has_node(i):
                G.add_node(i,time=i)
                G.add_edge(node,i)
                Q.put(i)
        if(Q.empty()):
            break
        node=Q.get()              
    return G  
def generatePath(G,goal):
    j=0
    flag=0
    l=list(nx.bfs_edges(G,(0,0))) #intial node = s = (0,0)
    for i in l: #return l value pass in this to generate graph 
       if(i[1]==goal):
          k=i[0]
          h=[]
          h.append(i)
          while(k!=(0,0)):
              if(l[j][1]==k):
                  k=l[j][0]
                  h.append(l[j])
              j=j-1
          flag=1    
          break    
       j=+1   
    if(flag!=0):
        h.reverse()
        print("You are able to fill JUG->1 and JUG->2 with", goal ,"amount of water at last \n")
        print("Filling Water Sequence be like : \n")
        print(h)
    else:
        print("You are not able to fill JUG->1 and JUG->2 with", goal ,"amount of water at last")

jug1=int(input("Enter JUG->1 Size in liter \n"))
jug2=int(input("Enter JUG->2 Size in liter \n"))
goal=[]
goal.append(int(input("Enter At The End JUG->1 Goal or You want to fill how much amount water at the last \n")))
goal.append(int(input("Enter At The End JUG->2 Goal or You want to fill how much amount water at the last \n")))
G=generateTree(jug1,jug2)
generatePath(G,tuple(goal)) 

