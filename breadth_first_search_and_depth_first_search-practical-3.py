#!/usr/bin/env python
# coding: utf-8

# In[10]:


def bfs(start, goal, dictionary):
    openNode = [(start,[start])]
    closed = []
    solutions = []

    while openNode != []:
        (node,path) = openNode.pop(0)

        if node == goal:
            solutions.append(path)

        allChilds = dictionary[node]

        if node not in closed:
            closed.append(node)

        testPath = []
        selectedPath = []

        for i in allChilds:
            testPath = path + [i]
            if len(testPath) == len(set(testPath)):
                selectedPath.append((i,testPath))

        openNode = selectedPath + openNode

    return solutions

if __name__ == "__main__":

    graph = {
        'R' : ['S', 'V'],
        'S' : ['R', 'W'],
        'V' : ['R'],
        'W' : ['S', 'X', 'T'],
        'T' : ['X', 'W'],
        'X' : ['W', 'T','Y'],
        'U' : ['T', 'Y'],
        'Y' : ['X', 'U'],   
        }

    
    a1=(input("Enter Starting Vertex in capital words like R , S ,T ,U ,V , W, X ,Y \n"))
    e1=(input("Enter Goal state in capital words like R , S ,T ,U ,V , W, X ,Y \n"))
    bfsAM = bfs(a1,e1,graph)
    print ("BFS Paths:[", a1 ,"->", e1 ,"]==>", bfsAM, "\n length:" ,len(bfsAM), "\n")


# In[12]:


def dfs(start, goal, dictionary):
    openNode = [(start,[start])]
    closed = []
    solutions = []

    while openNode != []:
        (node,path) = openNode.pop(0)

        if node == goal:
            solutions.append(path)

        allChilds = dictionary[node]

        if node not in closed:
            closed.append(node)

        testPath = []
        selectedPath = []

        for i in allChilds:
            testPath = path + [i]
            if len(testPath) == len(set(testPath)):
                selectedPath.append((i,path + [i]))

        openNode = openNode + selectedPath

    return solutions

if __name__ == "__main__":

    graph = {
        'V' : ['U','Y','X'],
        'W' : ['W','Z'],
        'X' : ['U','V','Y'],
        'U' : ['V','X'],
        'Y' : ['X','V','W'],
        'Z' : ['W']
        }

    
    a1=(input("Enter Starting Vertex in capital words like V,W,X,U,Y,Z \n"))
    e1=(input("Enter Goal state in capital words like V,W,X,U,Y,Z \n"))
    dfsAM = dfs(a1,e1,graph)
    print ("DFS Paths:[", a1 ,"->", e1 ,"]==>", dfsAM, "\n length:" ,len(dfsAM), "\n")


# In[ ]:





# In[ ]:




