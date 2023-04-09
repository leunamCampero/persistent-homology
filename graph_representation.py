import geopandas as gpd
import matplotlib.pyplot as plt
import momepy
import networkx as nx
from contextily import add_basemap
from libpysal import weights
from shapely.geometry import LineString



#df = gpd.GeoDataFrame(['a', 'b', 'c', 'd', 'e'], geometry=[l1, l2, l3, l4, l5])
bikes = gpd.read_file(r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\Cycle_paths.txt')
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
final_points = []
G = nx.Graph()
for i in coordn:
    final_points.append(i[0])
    final_points.append(i[len(i)-1])
    for j in range(len(i)-1):
        G.add_edge(i[j],i[j+1])

for i in list(G.nodes):
    if ((G.degree[i] == 2) and (i not in final_points)):
        e = [n for n in G.neighbors(i)]
        G.add_edge(e[0], e[1])
        G.remove_node(i)
for e in list(G.edges):
    if (e[0] == e[1]):
        G.remove_edge(*e)
positions = {n: [n[0], n[1]] for n in list(G.nodes)}

f, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
bikes.plot(color="k", ax=ax[0])
for i, facet in enumerate(ax):
    facet.set_title(("Streets", "Graph")[i])
    facet.axis("off")
nx.draw(G, positions, ax=ax[1], node_size=0.1)
plt.savefig('external.pdf', dpi=1200)
plt.show()
