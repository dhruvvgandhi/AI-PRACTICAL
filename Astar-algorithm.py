#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

#heuristic function
def Heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

#A-Star algorithm
def A_STAR(mainGraph1, source, goal):
    #openlist
    openList = [source]
    #list of visited nodes
    exploredList = []
    flag = 0
    #G(n) value
    gnValue = 0
    while openList:
        print("Open List",openList)
        #currentlt available node
        currentNode = openList.pop(0)
        if currentNode == goal:
            exploredList.append(currentNode)
            flag = 1
            break
        if currentNode not in exploredList:
            exploredList.append(currentNode)
            #finding adjacents
            adjacents = []
            #four possible direction of adjacents [----> UP - DOWN - LEFT - RIGHT <----]
            nei = [[0,1], [0, -1], [1, 0], [-1, 0]]
            #selecting appropriate adjacents for further exploration
            for i, j in nei:
                neb = [currentNode[0]+i, currentNode[1]+j]  
                if 0 <= neb[0] <= goal[0]:
                    if 0 <= neb[1] <= goal[1]:
                        if mainGraph[neb[0]][neb[1]] == 0:
                            adjacents.append(neb)
                    else:
                        continue
                else:
                    continue
            print("Adjacent Before", adjacents)
            #removing already explored nodes
            for i in adjacents:
                if i in exploredList:
                    adjacents.remove(i)
            print("Final Adjacents", adjacents)
            #H(n) values --- heuristic values
            localHeuristic = []
            #finding heuristic values and appending it to the list
            for i in range(len(adjacents)):
                localHeuristic.append(Heuristic(adjacents[i], goal)) 
            #F(n) values
            fnValue = []
            #F(n) = G(n) + H(n)
            for i in range(len(localHeuristic)):
                fnValue.append(localHeuristic[i] + gnValue)
            gnValue += 1
            print("Local F(n)", fnValue)
            localMin = min(fnValue)
            print("Local Minimum", localMin)
            for i in range(len(fnValue)):
                if fnValue[i] == localMin:
                    openList.append(adjacents[i])
                    break
                else:
                    pass 
            print("Current Path",exploredList)
            print("\n")
    if flag == 1:
        print("Successfully Reached to the Goal!!! ;)")
        print("Final Path :")
        for i in exploredList:
            print(i)
    else:
        print("Goal is Not Reachable :(")
#input source and goal from user
s = input("Enter Starting Node like [0 0]: ")
g = input("Enter Goal Node like [0 0]: ")
source = [int(i) for i in s.split()]
goal = [int(i) for i in g.split()]
#extracting total rows and column of the graph
row = (goal[0] - source[0]) + 1
column = (goal[1] - source[1]) + 1
#initializing main graph with zeros
mainGraph = [[0 for i in range(column)] for j in range(row)]
print("\n")
print("Simple Matrix Of  " + g )
print("\n")
for i in mainGraph:
    print(i)
print("\n")    
#inputing total number of obstacles from user
lObstacles = int(input("How Many Obstacles: "))
obstacles = []
#input position of obstacles, X and Y, one by one
for i in range(lObstacles):
    obstacle =  input("Enter Obstacle " + str(i+1) + " Node Position like [0 0] :")
    obstacles.append([int(i) for i in obstacle.split()])
#putting obstacles after in mainGraph
mainGraph1 = [[0 for i in range(column)] for j in range(row)]
for i, j in obstacles:
    mainGraph1[i][j] = 1
print("\n")
print("Matrix with Obstacles")
print("\n")
for i in mainGraph1:
    print(i)
print("\n")    
#calling A-Star algorithm
A_STAR(mainGraph1, source, goal)


# In[ ]:




