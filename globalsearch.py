import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
import itertools

def overlaps(t1, t2):
    a = t1[1] - t2[0]
    b = t1[1] - t1[0]
    return min(max(a, 0), b)

def connect_all_nodes(n, G):
    #connecting the current node to all the previous nodes if the 
    #overlap between two edges is not zero
    if n != 0:
        t1 = G.node[n]['edge_of_intgraph']
        for i in range(n):
            t2 = G.node[i]['edge_of_intgraph']
            overlap = overlaps(t1, t2)
            G.add_edge(i, n, weight = overlap)

def edge_graph(G):
    #generating the edge graph from the interaction graph
    #vertices represent the edge in the interaction graph
    #edges represent the overlaps between two different edges
    edgeGraph = nx.Graph()
    for i in range(G.number_of_edges()):
        e = np.array(G.edges())[i]
        edgeGraph.add_node(i, edge_of_intgraph = e)
        connect_all_nodes(i, edgeGraph)
    return edgeGraph
        
def gsearch(G):
    #finding the path traversing all the vertices which maximizes the sum of the weights
    s = 0
    maxS = 0
    rightorder = []
    pm = list(itertools.permutations(range(G.number_of_nodes())))
    for itm in range(len(pm)):
        pmt = pm[itm]
        for i in range(len(pmt))[0:-1]:
            s += G.edges[pmt[i],pmt[i+1]]['weight']
        if s >= maxS:
            maxS = s
            rightorder = list(pm[itm])
    return rightorder

def printrightop(G,rightorder):
    for i in rightorder:
        print(G.node[i]['edge_of_intgraph'])
    
