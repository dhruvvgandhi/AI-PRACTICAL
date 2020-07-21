#!/usr/bin/env python
# coding: utf-8

# In[6]:


class Graph:
    # Initialize the class
    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()
    def make_undirected(self):
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.graph_dict.setdefault(b, {})[a] = dist
    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        if not self.directed:
            self.graph_dict.setdefault(B, {})[A] = distance
    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)
class Node:
    def __init__(self, name:str, parent:str):
        self.name = name
        self.parent = parent
        self.g = 0 
        self.h = 0 
        self.f = 0 
    def __eq__(self, other):
        return self.name == other.name
    def __lt__(self, other):
         return self.f < other.f
    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))
def best_first_search(graph, heuristics, start, end):
    open = []
    closed = []
    start_node = Node(start, None)
    goal_node = Node(end, None)
    
    open.append(start_node)
    while len(open) > 0:
        open.sort()
        current_node = open.pop(0)
        closed.append(current_node)
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.name + ' : ' + str(current_node.g))
                path.append("==>")
                current_node = current_node.parent
            path.append(start_node.name + ' : ' + str(start_node.g))
            return path[::-1]
        neighbors = graph.get(current_node.name)
        for key, value in neighbors.items():
            neighbor = Node(key, current_node)
            if(neighbor in closed):
                continue
            neighbor.g = current_node.g + graph.get(current_node.name, neighbor.name)
            neighbor.h = heuristics.get(neighbor.name)
            neighbor.f = neighbor.h
            if(add_to_open(open, neighbor) == True):
                open.append(neighbor)
    return None
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f >= node.f):
            return False
    return True
def main():
    graph = Graph()
    graph.connect('A', 'B', 11)
    graph.connect('A', 'D', 7)
    graph.connect('A', 'C', 14)
    graph.connect('B', 'E',15)
    graph.connect('C', 'E', 8)
    graph.connect('C', 'F', 10)
    graph.connect('D', 'F', 25)
    graph.connect('E', 'H', 9)
    graph.connect('H', 'G', 10)
    graph.connect('F', 'G', 20)
    graph.connect('G', 'I', 10)
    graph.connect('G', 'K', 10)
    graph.connect('I', 'J', 50)
    graph.connect('K', 'J', 30)
    graph.make_undirected()
    heuristics = {}
    heuristics['A'] = 40
    heuristics['B'] = 32
    heuristics['C'] = 25
    heuristics['D'] = 35
    heuristics['E'] = 19
    heuristics['F'] = 17
    heuristics['G'] = 0
    heuristics['I'] = 8
    heuristics['H'] = 10
    heuristics['J'] = 41
    heuristics['K'] = 10
    a1=(input("Enter Starting Vertex in capital words like A,B,C,D,F,G,H,I,J,K \n"))
    e1=(input("Enter Goal state in capital words like A,B,C,D,F,G,H,I,J,K \n"))
    path = best_first_search(graph, heuristics,a1, e1)
    print(path)
    print()
# Tell python to run main method
if __name__ == "__main__": main()


# In[ ]:




