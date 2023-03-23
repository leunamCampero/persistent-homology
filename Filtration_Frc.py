import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

clear = lambda: os.system('cls')
clear()

#Function Filtration
def filtration(RIN, RIE, RGN, alp):
    '''
    ----------
     Parameters
     ----------
     RIN: numpy_array
         The intersection graph nodes database

     RIE: numpy_array
         The intersection graph edges database

     RGN: numpy_array
         The road graph nodes database

     alp: float
          Lower bound to define the filtration
     --------
     Returns:
     --------
     G: networkx DiGraph
         A digraph filtration.
    '''
    SG = nx.DiGraph()
    id_f = [int(RGN[i, 0]) for i in range(RGN.shape[0]) if RGN[i, 9] <= alp]
    ind_f = [list(RGN[:, 0]).index(i) for i in id_f]
    RIEf = RIE[ind_f]
    SG.add_edges_from([(int(RIEf[i, 0]), int(RIEf[i, 1])) for i in range(RIEf.shape[0])])
    largest = list(max(nx.strongly_connected_components(SG), key=len))
    LSCS = SG.subgraph(largest)
    pv = [421, 422, 423, 424, 425, 426, 427, 428]
    v_colors = ['c', 'y', 'k', 'b', 'm', 'r']
    frc_alp = [1, 3, 4, 5, 6, 7]
    for k in frc_alp:
        SG = nx.DiGraph()
        id_f = [int(RGN[i, 0]) for i in range(RGN.shape[0]) if RGN[i, 9] == k]
        ind_f = [list(RGN[:, 0]).index(i) for i in id_f]
        RIEf = RIE[ind_f]
        SG.add_edges_from([(int(RIEf[i, 0]), int(RIEf[i, 1])) for i in range(RIEf.shape[0]) if ((int(RIEf[i, 0]) in largest) and (int(RIEf[i, 1]) in largest))])
        # if ((int(RIEf[i, 0]) in largest) and (int(RIEf[i, 1]) in largest))
        nSG = sorted([i - 1 for i in list(SG.nodes)])
        RINf = RIN[nSG]
        npos = {}
        for x in range(RINf.shape[0]):
            npos[nSG[x] + 1] = (RINf[x, 0], RINf[x, 1])
        if k != 1:
            plt.subplot(pv[k - 2])
            nx.draw_networkx(SG, pos=npos, with_labels=False, arrows=True, node_size=5, edge_color=v_colors[k - 2],
                             arrowsize=7.5, arrowstyle='-|>')
            plt.title(("FRC", k))
            plt.subplot(pv[6])
            nx.draw_networkx(SG, pos=npos, with_labels=False, arrows=True, node_size=5, edge_color=v_colors[k - 2],
                             arrowsize=7.5, arrowstyle='-|>')
            plt.title(("ALL_FRC"))
        else:
            plt.subplot(pv[k - 1])
            nx.draw_networkx(SG, pos=npos, with_labels=False, arrows=True, node_size=5, edge_color=v_colors[k - 1],
                             arrowsize=7.5, arrowstyle='-|>')
            plt.title(("FRC", k))
            plt.subplot(pv[6])
            nx.draw_networkx(SG, pos=npos, with_labels=False, arrows=True, node_size=5, edge_color=v_colors[k - 1],
                             arrowsize=7.5, arrowstyle='-|>')
            plt.title(("ALL_FRC"))
    plt.show()









#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#GrenobleNetwork_RoadGraph
#Create the csv files.
#ReadGNRGNodes = pd.read_excel (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_RoadGraph_Nodes.xlsx')
#ReadGNRGEdges = pd.read_excel (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_RoadGraph_Edges.xlsx')
#ReadGNRGNodes.to_csv (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_RoadGraph_Nodes.csv', index = None, header=True)
#ReadGNRGEdges.to_csv (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_RoadGraph_Edges.csv', index = None, header=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# GrenobleNetwork_IntersectionGraph
#Create the csv files.
ReadGNIGNodes = pd.read_excel (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_IntersectionGraph_Nodes.xlsx')
ReadGNIGEdges = pd.read_excel (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_IntersectionGraph_Edges.xlsx')
ReadGNIGNodes.to_csv (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_IntersectionGraph_Nodes.csv', index = None, header=True)
ReadGNIGEdges.to_csv (r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_IntersectionGraph_Edges.csv', index = None, header=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Call the csv files to a panda frame.
# GrenobleNetwork_IntersectionGraph
RINpanda = pd.read_csv(r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_IntersectionGraph_Nodes.csv')
RIEpanda = pd.read_csv(r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_IntersectionGraph_Edges.csv')

#GrenobleNetwork_RoadGraph
RGNpanda = pd.read_csv(r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_RoadGraph_Nodes.csv')
RGEpanda = pd.read_csv(r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\GrenobleNetwork_RoadGraph_Edges.csv')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Made the panda frames into numpy arrays
# GrenobleNetwork_IntersectionGraph
RIN = RINpanda.to_numpy()
RIE = RIEpanda.to_numpy()
#GrenobleNetwork_RoadGraph
RGN = RGNpanda.to_numpy()
RGE = RGEpanda.to_numpy()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Test filtration function
alp = 8
s = filtration(RIN, RIE, RGN, alp)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the Graph.
# G = nx.DiGraph()
# nodes = [int(i+1) for i in range(RIN.shape[0])]
# G.add_edges_from([(int(RIE[i,0]),int(RIE[i,1])) for i in range(RIE.shape[0])])
# npos = {}
# for x in range(RIN.shape[0]):
#     npos[int(x+1)] = (RIN[x,0],RIN[x,1])
# nx.draw_networkx(G, pos = npos, with_labels=False, arrows = True, node_size=5, arrowsize=7.5, arrowstyle='-|>')
# plt.savefig('GrenobleNetwork_IntersectionGraph.pdf', dpi=1200)
# plt.show()

