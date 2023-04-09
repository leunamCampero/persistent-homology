import geopandas as gpd
import matplotlib.pyplot as plt
import momepy
import networkx as nx
from contextily import add_basemap
from libpysal import weights
from shapely.geometry import LineString
import pandas as pd

def duplicate_d(dw):
    iterator_k = list(dw.keys())
    s = set()
    for i in list(iterator_k):
        s.add(i[0])
        s.add(i[1])
    ddw = {}
    for i in list(s):
        for j in list(s):
            if ((i,j) in iterator_k):
                ddw[(i,j)] = dw[(i,j)]
            if ((j, i) in iterator_k):
                ddw[(i, j)] = dw[(j, i)]
        ddw[i,i] = 0
    return ddw

#df = gpd.GeoDataFrame(['a', 'b', 'c', 'd', 'e'], geometry=[l1, l2, l3, l4, l5])
bikes = gpd.read_file(r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\Cycle_paths.txt')
df1 = pd.DataFrame(bikes)
list(bikes['geometry'][0].coords)
#bikes.plot(figsize=(10, 10)).set_axis_off()
#bikes = momepy.extend_lines(bikes, 0.001)
#bikes_e = momepy.extend_lines(bikes, 0.0000)
#bikes_extended.plot(figsize=(10, 10)).set_axis_off()
#bikes_e.geometry = momepy.close_gaps(bikes_e, 0.000)
#bikes_e = momepy.remove_false_nodes(bikes)
#bikes = momepy.extend_lines(bikes, 1)
def coords(geom):
    return list(geom.coords)
coords = bikes.apply(lambda row: coords(row.geometry), axis=1)
coordn = coords.to_numpy()
pandadata = pd.DataFrame(bikes)
numpydata = pandadata.to_numpy()
final_points = []
G = nx.Graph()
c = 0
dw = {}
for i in coordn:
    final_points.append(i[0])
    final_points.append(i[len(i)-1])
    for j in range(len(i)-1):
        if (numpydata[c][2] == 'chronovelo'):
            G.add_edge(i[j],i[j+1], weight = 1)
            dw[(i[j],i[j+1])] = 1
            dw[(i[j + 1], i[j])] = 1
        if (numpydata[c][2] == 'voieverte'):
            G.add_edge(i[j],i[j+1], weight = 2)
            dw[(i[j], i[j + 1])] = 2
            dw[(i[j + 1], i[j])] = 2
        if (numpydata[c][2] == 'veloamenage'):
            G.add_edge(i[j],i[j+1], weight = 3)
            dw[(i[j], i[j + 1])] = 3
            dw[(i[j + 1], i[j])] = 3
        if (numpydata[c][2] == 'velononamenage'):
            G.add_edge(i[j],i[j+1], weight = 4)
            dw[(i[j], i[j + 1])] = 4
            dw[(i[j + 1], i[j])] = 4
        if (numpydata[c][2] == 'velodifficile'):
            G.add_edge(i[j],i[j+1], weight = 5)
            dw[(i[j], i[j + 1])] = 5
            dw[(i[j + 1], i[j])] = 5
    c = c + 1
for i in list(G.nodes):
    nh = [n for n in G.neighbors(i)]
    if ((len(nh) == 2) and (i not in final_points)):
        G.add_edge(nh[0], nh[1], weight = dw[(i,nh[0])])
        G.remove_node(i)
for e in list(G.edges):
    if (e[0] == e[1]):
        G.remove_edge(*e)
pos = {n: [n[0], n[1]] for n in list(G.nodes)}

f, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
bikes.plot(color="k", ax=ax[0])
for i, facet in enumerate(ax):
    facet.set_title(("Bikes", "Graph")[i])
    facet.axis("off")
s1 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 1]
s2 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 2]
s3 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 3]
s4 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 4]
s5 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 5]

# edge weight labels

#plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=2)
nx.draw_networkx_edges(G, pos, edgelist=s1, width=2, edge_color="b")
nx.draw_networkx_edges(G, pos, edgelist=s2, width=2, edge_color="r")
nx.draw_networkx_edges(G, pos, edgelist=s3, width=2, edge_color="y")
nx.draw_networkx_edges(G, pos, edgelist=s4, width=2, edge_color="m")
nx.draw_networkx_edges(G, pos, edgelist=s5, width=2, edge_color="g")

# node labels
#nx.draw_networkx_edge_labels(G, pos, font_size=0.1, font_family="sans-serif")
#edge_labels = nx.get_edge_attributes(G, "weight")
#nx.draw_networkx_edge_labels(G, pos, edge_labels, alpha=0.4)

#nx.draw_networkx_nodes(
#    G,
#    pos,
#    nodelist=list(p.keys()),
#    node_size=80,
#    node_color="k"
    # node_color=list(p.values()),
    # cmap=plt.cm.Blues,
    # cmap=plt.cm.Reds_r,
    #cmap=plt.cm.Reds,
#)
nx.draw(G, pos, ax=ax[1], node_size=0.1)
plt.savefig('external.pdf', dpi=1200)
plt.show()
