import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix


clear = lambda: os.system('cls')
clear()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
class SCMaximalFacets:
    """
    A class used to create a simplex.

    ----------
    Attributes
    ----------
    v: list
        A list of the maximal faces of the simplicial complex.
    """

    def __init__(self, v):
        '''
        Generate the nodes of a simplicial complex.

        ----------
        Attributes
        ----------
        v: list
            A list of the maximal faces of the simplicial complex.

        -------
        Raises
        -------
        AttributeError:
            If the maximal faces are not immutable objects.

        ---------
        Examples:
        ---------
        To make a simplicial complex associated with a graph,
        use the ``SimplicialComplex`` class. We need a graph G.
            >>> sc = SCMaximalFacets([[0,1,2,3],[3,4,5]])
            >>> print(sc.SCverices)
            [0, 1, 2, 3, 4, 5]

        '''
        #self.v = v
        self.SCFaset = v
        #self.SCvertices = set()
        #for x in self.v:
        #    self.SCFaset.append(x)
         #   for y in x:
          #      self.SCvertices.add(y)
        #self.SCvertices = list(self.SCvertices)

    def faces(self):
        """
        Makes the faces of a simplicial complex.

           A simplicial complex SC is uniquely determined by its maximal faces,
           and it must contain every face of a simplex, also the intersection
           of any two simplexes of SC is a face of each of them.

        ----------
        Returns
        ----------
        list:
            A list of the faceset of a simplicial complex.

        ---------
        Examples:
        ---------
        To create the faces of a simplical complex use,
        ``SimplicialComplex.faces()''.
            >>> sc = SCMaximalFacets([[0,1,2,3],[3,4,5]])
            >>> print(sc.faces())
            [(1, 3), (0, 1, 2), (0, 1, 3), (0, 3), (1, 2), (1,), (0, 2, 3), (3,), (5,), (0, 1, 2, 3), (4, 5), (1, 2, 3), (2, 3), (3, 5), (0, 1), (0,), (2,), (4,), (), (3, 4, 5), (3, 4), (0, 2)]

        .. Note:: The faces are sorted by their dimension.

            """
        ll = []
        for i in self.SCFaset:
            ll.append(tuple(sorted(i)))
        return ll

    def p_simplices_2d(self):
        """
        Creates a list of the faces of a simplex with dimension p.

        ----------
        Parameters
        ----------
        p: int
            The dimension of the desired faces.

        --------
        Returns:
        --------
        list:
            A list of the faces of a simplex with dimension p.

        ---------
        Examples:
        ---------
        The p-simplices are done with
        "SimplicialComplex.p_simplex(p)".
            >>> sc = SCMaximalFacets([[0,1,2]])
            >>> print(sc.faces())
            [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
            >>> print(sc.p_simplices(0))
            [(0,), (1,), (2,)]
            >>> print(sc.p_simplices(2))
            [(0, 1, 2)]
            >>> print(sc.p_simplices(1))
            [(0, 1), (0, 2), (1, 2)]
            >>> print(sc.p_simplices(5))
            []

        .. Note:: If there are not faces of dimension p,
        the method return a empty list like in
        ``sc.p_simplex(5)''.

        """
        return list(filter(lambda face: (len(face) <= 3), self.faces()))

    def p_simplices(self, p):
        """
        Creates a list of the faces of a simplex with dimension p.

        ----------
        Parameters
        ----------
        p: int
            The dimension of the desired faces.

        --------
        Returns:
        --------
        list:
            A list of the faces of a simplex with dimension p.

        ---------
        Examples:
        ---------
        The p-simplices are done with
        "SimplicialComplex.p_simplex(p)".
            >>> sc = SCMaximalFacets([[0,1,2]])
            >>> print(sc.faces())
            [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
            >>> print(sc.p_simplices(0))
            [(0,), (1,), (2,)]
            >>> print(sc.p_simplices(2))
            [(0, 1, 2)]
            >>> print(sc.p_simplices(1))
            [(0, 1), (0, 2), (1, 2)]
            >>> print(sc.p_simplices(5))
            []

        .. Note:: If there are not faces of dimension p,
        the method return a empty list like in
        ``sc.p_simplex(5)''.

        """
        return list(filter(lambda face: (len(face) == p + 1), self.p_simplices_2d()))

    def dimension(self):
        """
        Gives the dimension of a simplicial complex.
        ----------
        Parameters
        ----------
        p: int
            The dimension of the faces.

        --------
        Returns:
        --------
        (a - 1): int
            The dimension of the simplicial complex.
        -------
        Raises:
        -------
            Return ``-1'' if the simplicial complex is empty.

        Examples:
        To use the method dimension write
        ``SimplicialComplex.dimension()''.
            >>> sc = SCFromFacets([[0,1,2,4]])
            >>> print(sc.dimension())
            3
        """
        a = 0
        for x in self.p_simplices_2d():
            if (len(x) > a):
                a = len(x)

        return a-1

    def s_dimension(self):
        """
        Creates a dictionary of the dimension of every element in the Simplicial Complex.

        --------
        Returns:
        --------
        dd: dictionary
            Keys as the simplices, and values as their respectively dimension.

        ---------
        Examples:
        ---------
            >>> sc = SCMaximalFacets([[0,1,2]])
            >>> print(sc.s_dimension())
            {(0,):0, (1,):0, (2,):0, (0, 1):1, (0, 2):1, (1, 2):1, (0, 1, 2):2]

        """
        dd = {}
        for i in range(self.dimension()+1):
            for j in self.p_simplices(i):
                dd[j] = len(j)-1
        return dd

    def boundary(self):
        """
        Gives a dictionary of the boundary operator on all the elements of the simplicial complex

        --------
        Returns:
        --------
        bd: dictionary
            Keys are the simplices and the values are themselves under the boundary operator

        -------
        Raises:
        -------
        AttributeError:
            If p is lower than zero or bigger than the dimension of the
            simplicial complex dimension.

        ---------
        Examples:
        ---------
        To create a basis for the group of oriented p-chains, use
        ``SimplicialComplex.basis_group_oriented_p_chains(p)``.
            >>> sc = SCMaximalFacets([[0,1,2,3],[3,4,5]])
            >>> print(sc.boundary())
            {(2,): [], (5,): [], (4,): [], (1,): [], (0,): [], (3,): [],
            (3, 4): [(4,), (3,)], (0, 2): [(2,), (0,)], (1, 3): [(3,), (1,)], (4, 5): [(5,), (4,)],
            (0, 1): [(1,), (0,)], (1, 2): [(2,), (1,)], (3, 5): [(5,), (3,)],
            (0, 3): [(3,), (0,)], (2, 3): [(3,), (2,)], (0, 1, 3): [(1, 3), (0, 3), (0, 1)],
            (0, 1, 2): [(1, 2), (0, 2), (0, 1)], (0, 2, 3): [(2, 3), (0, 3), (0, 2)],
            (1, 2, 3): [(2, 3), (1, 3), (1, 2)], (3, 4, 5): [(4, 5), (3, 5), (3, 4)], (0, 1, 2, 3): [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]}
        """
        bd = {}
        for i in range(self.dimension() + 1):
            for j in self.p_simplices(i):
                k = self.s_dimension()[j]
                if (k == 0):
                    bd[j]=[]
                else:
                    av = []
                    for c in range(self.s_dimension()[j]+1):
                        hl = []
                        for h in j:
                            hl.append(h)
                        hl.pop(c)
                        av.append(tuple(hl))
                    bd[j] = av
        return bd


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def SCG(G):
    '''
    A simplicial complex is associated with a graph by means of its maximal cliques.
    '''
    v = []
    for x in list(nx.enumerate_all_cliques(G)):
        w = []
        for y in x:
            w.append(y)
        v.append(w)
    return SCMaximalFacets(v)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def sub_lists(my_list):
    """
    A function that find all the sublists of a list given.
    """
    subs = []
    for i in range(0, len(my_list)+1):
        temp = [list(x) for x in combinations(my_list, i)]
        if len(temp)>0:
           subs.extend(temp)
    return subs

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def eq_elements(a, b):
    """
    A function that identify when tuples are equal except by orientation.

    ----------
    Parameters
    ----------
    a: tuple
        The first tuple.
    b: tuple
        The second tuple.

    --------
    Returns:
    --------
    bool:
        True if the tuples are equal except by orientation,
        False otherwise.

    -------
    Raises:
    -------
    TypeError:
        If the tuples don't have the same structure, for
        example:
            a = ((0,1),(2,3),(5,6))
            b = ((1,0))
    ---------
    Examples:
    --------
    To see if two tuples are equal use ``eq_elements''.
        >>> a1 = ((0,1),(2,3),(5,6))
        >>> b1 = ((0,3),(2,1),(5,6))
        >>> a2 = ((0,1),(2,3),(5,6))
        >>> b2 = ((6,5),(1,0),(3,2))
        >>> print(eq_elements(a1,b1))
        False
        >>> print(eq_elements(a2,b2))
        True
    """
    if ((isinstance(a, int) == True) or (isinstance(a, str) == True)):
        return a == b
    if ((isinstance(a[0], int) == True) or (isinstance(a[0], str) == True)):
        return (set() == set(a).difference(set(b)))
    else:
        for i in range(len(a)):
            test = False
            for j in range(len(b)):
                if (eq_elements(a[i], b[j]) == True):
                    test = True
            if (test == False):
                return False
        else:
            return True

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def tuple_sorted(a):
    """
    Sorted tuples of tuples.

    ----------
    Parameters
    ----------
    a: tuple
        A tuple the which will be sorted.

    --------
    Returns:
    --------
    tuple:
        The tuple sorted.

    ---------
    Examples:
    ---------
    The function ``sorted'' don't sort tuples of tuples, but
    this function can do it, below is showing examples of
    both functions:
        >>> a1 = ((6,5),(1,0),(3,2))
        >>> a2 = (((4,2),(1,0),(5,3)),((2,3),(1,0),(6,4)))
        >>> print(sorted(a1))
        [(1, 0), (3, 2), (6, 5)]
        >>> print(tuple_sorted(a1))
        ((0, 1), (2, 3), (5, 6))
        >>> print(sorted(a2))
        [((2, 3), (1, 0), (6, 4)), ((4, 2), (1, 0), (5, 3))]
        >>> print(tuple_sorted(a2))
        (((0, 1), (2, 3), (4, 6)), ((0, 1), (2, 4), (3, 5)))

    """
    if ((isinstance(a, int) == True) or (isinstance(a, str) == True)):
        return a
    if ((isinstance(a[0], int) == True) or (isinstance(a[0], str) == True)):
        return sorted(a)
    else:
        w = []
        for b in a:
            w.append(tuple(tuple_sorted(b)))
        return tuple(sorted(tuple(w)))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def scdiff(s2, s1):
    """
    Returns a list with the difference of the simplicial complexes, sorted by dimension.
    """
    v2 = s2.p_simplices_2d()
    v1 = s1.p_simplices_2d()
    v3 = list(set(v2)-set(v1))
    return sort_ln(v3)#sorted(v3, key=len)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def sort_ln(v):
    """
    Returns a list sorted by dimension and numbers.
    """
    D = {}
    D[1] = []
    D[2] = []
    D[3] = []
    for i in v:
        D[len(i)].append(i)
    svp = []
    for i in list(D.keys()):
        for j in sorted(D[i]):
            svp.append(j)
    return svp

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def boundary_matrix(sf):
    """
    Creates the boundary matrix of the simplicial filtration
    ----------
    Parameters
    ----------
    sf: list
        simplicial complex filtration list
    --------
    Returns:
    --------
    M: matrix
        The reduced boundary matrix
    pi: dictionary
        Keys the connected components and cycles that are not born or died in the same stage.
    -------
    """
    sv = []
    n = len(sf)
    sc = sf[n - 1]
    dv = []
    for i in sf[0].p_simplices_2d():
        if (i != ()):
            sv.append(i)
    sv = sorted(sv)
    dv.append(0)
    dv.append(len(sv))
    for i in range(n - 1):
        for j in scdiff(sf[i + 1], sf[i]):
            sv.append(j)
        dv.append(len(sv))
    nf = len(sv)
    M = csr_matrix((nf, nf), dtype=np.int8).toarray()
    dsc = sc.boundary()
    # sv = [(1,), (2,), (3,), (4,), (1, 3), (2, 3), (1, 2), (2, 4), (1, 2, 3), (1, 4), (3, 4), (1, 2, 4), (1, 3, 4), (2, 3, 4) ]
    osv = []
    for i in sv:
        osv.append([i])
    for i in range(nf):
        if (len(sv[i]) > 1):
            for j in dsc[sv[i]]:
                M[sv.index(j), i] = 1
    # M =  np.array([[0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # nf = M.shape[0]
    for i in range(nf):
        if (np.count_nonzero(M[:, i]) > 0):
            # print(i, nf)
            nz = np.max(np.nonzero(M[:, i]))
            eo = 0
            for j in range(i):
                if (np.count_nonzero(M[:, j]) > 0):
                    if (np.max(np.nonzero(M[:, j])) == nz):
                        eo = j
            c = 0
            while (eo > 0):
                print(i, nf, c)
                M[:, i] = M[:, i] + M[:, eo]
                osv[i].append(sv[eo])
                for j in range(nf):
                    M[j, i] = M[j, i] % 2
                if (np.count_nonzero(M[:, i]) > 0):
                    nz = np.max(np.nonzero(M[:, i]))
                    eo = 0
                    for j in range(i):
                        if (np.count_nonzero(M[:, j]) > 0):
                            if (np.max(np.nonzero(M[:, j])) == nz):
                                eo = j
                else:
                    eo = 0
                c = c + 1
    pi = {}
    for i in range(nf):
        if (np.count_nonzero(M[:, i]) == 0):
            if (np.count_nonzero(M[i, :]) != 0):
                fo = np.min(np.nonzero(M[i, :]))
                fz = np.max(np.nonzero(M[:, fo]))
                if (i < fz):
                    av1 = fin_d(i, dv)
                    av2 = dv[len(dv) - 1] - 1
                    av2 = fin_d(av2, dv)
                    if (av2 - av1 > 0):
                        pi[tuple(osv[i])] = [av1, av2]
                else:
                    av1 = fin_d(i, dv)
                    av2 = fin_d(fo, dv)
                    if (av2 - av1 > 0):
                        pi[tuple(osv[i])] = [av1, av2]
            else:
                av1 = fin_d(i, dv)
                av2 = dv[len(dv) - 1] - 1
                av2 = fin_d(av2, dv)
                if (av2 - av1 > 0):
                    pi[tuple(osv[i])] = [av1, av2]

    return M, pi


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def fin_d(fo, dv):
    """
    Function that determines in which age of the persistent homology in an element
    --------
    Returns:
    --------
    sf: set
        Collection of the simplicial filtration
    -------
    """
    for j in range(len(dv)-1):
        if (dv[j] <= fo < dv[j+1]):
            return j

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_pd(M, pi):
    """
    Plot the persistent homology bar diagram
    ----------
    Parameters
    ----------
    M: matrix
        The reduced boundary matrix
    pi: dictionary
        Keys the connected components and cycles that are not born or died in the same stage.
    --------
    Returns:
    --------
    plot
        Persitent homology diagram plot
    -------
    """
    abz = []
    bbz = []
    abo = []
    bbo = []
    lk = list(pi.keys())
    lv = list(pi.values())
    for i in range(len(lv)):
        if (len(lk[i][0]) == 1):
            abz.append(lv[i][0])
            bbz.append(lv[i][1]-lv[i][0])
        else:
            abo.append(lv[i][0])
            bbo.append(lv[i][1]-lv[i][0])
    ind_k = list(pi.keys())[0:len(abz)]
    df = pd.DataFrame({'Before born': abz,
                       'Lifespan': bbz}, index=ind_k)
    # axes = df.plot.bar(
    #     rot=0, subplots=True, color={"Before born": "red", "Lifespan": "green"}
    # )
    # axes[1].legend(loc=2)
    # ax = df.plot.bar(rot=0)
    ax = df.plot.barh(stacked=True, color={"Before born": "white", "Lifespan": "blue"})
    #plt.show()
    #-------------------------------------------------------------------------------
    ind_k = list(pi.keys())[len(abz):len(lv)]
    df = pd.DataFrame({'Before born': abo,
                       'Lifespan': bbo}, index=ind_k)
    # axes = df.plot.bar(
    #     rot=0, subplots=True, color={"Before born": "red", "Lifespan": "green"}
    # )
    # axes[1].legend(loc=2)
    # ax = df.plot.bar(rot=0)
    ax = df.plot.barh(stacked=True, color={"Before born": "white", "Lifespan": "red"})
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Function Filtration
def lscsg_ww(RIN, RIE, RGN, alp):
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
         The largest strongly connected subgraph
     weights: dictionary
         Keys are the edges and the values is the functional road classification of the corresponding edge.
     pos = dictionary
         normalized coordinates in the plane of the nodes list
    '''
    SG = nx.DiGraph()
    id_f = [int(RGN[i,0]) for i in range(RGN.shape[0]) if RGN[i,9] <= alp]
    ind_f = [list(RGN[:,0]).index(i) for i in id_f]
    RIEf = RIE[ind_f]
    SG.add_edges_from([(int(RIEf[i, 0]), int(RIEf[i, 1])) for i in range(RIEf.shape[0])])
    nSG = sorted([i-1 for i in list(SG.nodes)])
    RINf = RIN[nSG]
    largest = max(nx.strongly_connected_components(SG), key=len)
    lscs = SG.subgraph(list(largest))
    el_l = list(lscs.edges)
    e_sg = list(SG.edges)
    ei_l = [i for i in range(len(el_l)) if el_l[i] in e_sg]
    lRGN = RGN[ei_l]
    fG = nx.DiGraph()
    weights = {}
    fG.add_weighted_edges_from([(el_l[i][0], el_l[i][1], lRGN[i,9]) for i in range(len(el_l))])
    for i in range(len(el_l)):
        weights[el_l[i]] = lRGN[i,9]
    npos = {}
    for x in range(RINf.shape[0]):
        if ((x+1) in largest):
            npos[nSG[x]+1] = (RINf[x, 0], RINf[x, 1])
    return fG, weights, npos


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Function minimal distance according to the functional road classification
def dist(G, W):
    '''
    ----------
     Parameters
     ----------
     G: networkx DiGraph
        Graph

     W: dictionary
        Weights of the edges

     --------
     Returns:
     --------
     Dd: dictionary
         keys as the origin destination nodes and values as the minimum sum values over all the paths
    '''
    nv = list(G.nodes)
    p = {}
    #print(list(nx.all_simple_paths(G, source=1, target=2)))
    for i in nv:
        for j in nv:
            if (i != j):
                p[(i,j)] = nx.dijkstra_path(G, i, j)
    Dd = []
    for i in nv:
        for j in nv:
            if (i != j):
                s = 0
                for k in range(len(p[(i,j)])-1):
                    s = s + W[(p[(i,j)][k],p[(i,j)][k+1])]
            else:
                s = 0
            Dd.append([i,j,s])
    Dd = np.array(Dd)
    np.savetxt("distances.csv", Dd, delimiter = ",")
    return Dd

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def sym_d(Md, vn):
    '''
    ----------
     Parameters
     ----------
     Md: numpy.array
        Array where the fist, second and third column are the origin node, destination node and distance respectively
     vn: list
        list of graph nodes
     --------
     Returns:
     --------
     s_m: dictionary
         keys as the origin-destination nodes and values as the symmetrization distance with sign -
     s_p: dictionary
         keys as the origin-destination nodes and values as the symmetrization distance with sign +
    '''
    ad = {}
    for i in range(Md.shape[0]):
        ad[(int(Md[i,0]),int(Md[i,1]))] = Md[i,2]
    s_m = {}
    s_p = {}
    for i in vn:
        for j in vn:
            s_m[(i,j)] = min(ad[(i,j)],ad[(j,i)])
            s_p[(i, j)] = max(ad[(i, j)], ad[(j, i)])
    return s_m, s_p

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def graph_filtration(s_m, G, eps):
    '''
    ----------
    Parameters
    ----------
    s_m: dictionary
        keys as the origin-destination nodes and values as the symmetrization distance with sign -
    G: nx.DiGraph
        Graph to be analized
    eps:
        Upper bound to create the filtration
    --------
    Returns:
    --------
    Gf: nx.Graph
        A graph filtration corresponding to the eps (epsilon) parameter
    '''
    Gf = nx.Graph()
    for i in list(G.edges):
        if(s_m[i] <= eps):
            Gf.add_edge(*i)
    return Gf

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def filtration_graph_list(s_m, G):
    '''
    ----------
    Parameters
    ----------
    s_m: dictionary
         keys as the origin-destination nodes and values as the symmetrization distance with sign -
    G: nx.DiGraph
        Graph to be analized
    --------
    Returns:
    --------
    vGf: list
        list from all graph filtration from epsilon parameter 1 to max s_m.values()
    '''
    vGf = []
    Gf = nx.DiGraph()
    Gf.add_nodes_from(list(G.nodes))
    vGf.append(Gf)
    for eps in range(1, 7):
        Gf = nx.DiGraph()
        for i in list(G.edges):
            if (s_m[i] <= eps):
                Gf.add_edge(*i)
        vGf.append(Gf)
    return vGf

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def filtration_graph_list_p(s_m, G):
    '''
    ----------
    Parameters
    ----------
    s_m: dictionary
         keys as the origin-destination nodes and values as the symmetrization distance with sign -
    G: nx.DiGraph
        Graph to be analized
    --------
    Returns:
    --------
    vGf: list
        list with the graph filtration from epsilon parameter 1 to max s_m.values()
    '''
    vGf = []
    Gf = nx.Graph()
    Gf.add_nodes_from(list(G.nodes))
    vGf.append(Gf)
    # for eps in range(1, int(max(list(s_m.values())))):
    for eps in range(1, 8):
        Gf = Gf.copy()
        for e in list(G.edges):
            if (0 < s_m[e] <= eps):
                Gf.add_edge(e[0], e[1])
        vGf.append(Gf)
    return vGf

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def filtration_sc(vGf):
    '''
    ----------
    Parameters
    ----------
    vGf: list
        list with the graph filtration from epsilon parameter 1 to max s_m.values()
    --------
    Returns:
    --------
    scGf: list
        list with the simplicial complexes associated with the graph filtration from epsilon parameter 1 to max s_m.values()
    '''
    l = [SCG(i) for i in vGf]
    return l




#-------------------------------------------------------------------------------------------------------------------------------------------------------------

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
G, W, pos = lscsg_ww(RIN, RIE, RGN, alp)
distances = pd.read_csv(r'C:\Users\ali_saleme\Documents\Phd_Inria\PyCharm\venv\distances.csv')
Md = distances.to_numpy()
s_m, s_p = sym_d(Md, list(G.nodes))
G1 = nx.Graph()
G1.add_edges_from(list(G.edges))
sce = SCG(G1)
vGf= filtration_graph_list_p(s_m, G)
sf = filtration_sc(vGf)
M, pi = boundary_matrix(sf)
plot_pd(M, pi)
#graph_filtration(s_m, G, eps)
#sol = dist(G, W)
#print(sol)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the Graph.
# G = nx.DiGraph()
# nodes = [int(i+1) for i in range(RIN.shape[0])]
# G.add_edge_from([(int(RIE[i,0]),int(RIE[i,1])) for i in range(RIE.shape[0])])
# npos = {}
# for x in range(RIN.shape[0]):
#     npos[int(x+1)] = (RIN[x,0],RIN[x,1])
# nx.draw_networkx(G, pos = npos, with_labels=False, arrows = True, node_size=5, arrowsize=7.5, arrowstyle='-|>')
# plt.savefig('GrenobleNetwork_IntersectionGraph.pdf', dpi=1200)
# plt.show()

