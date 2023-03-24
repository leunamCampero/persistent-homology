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


class SCset:
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
        self.v = v

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
        sv = []
        for i in self.v:
            if len(i) != 0:
                sv.append(tuple(tuple_sorted(i)))
            else:
                sv.append(i)
        return sv

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
        return list(filter(lambda face: (len(face) == p + 1), self.faces()))

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
        for x in self.faces():
            if (len(x) > a):
                a = len(x)
        return a - 1

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
        self.v = v
        self.SCFaset = []
        self.SCvertices = set()
        for x in self.v:
            self.SCFaset.append(x)
            for y in x:
                self.SCvertices.add(y)
        self.SCvertices = list(self.SCvertices)

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

        faceset = set()
        for faset in self.SCFaset:
            for face in sub_lists(faset):
                if len(face) != 0:
                    faceset.add(tuple(tuple_sorted(face)))
                else:
                    faceset.add(tuple(face))
        return list(faceset)

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
        return list(filter(lambda face: (len(face) == p + 1), self.faces()))

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
        for x in self.faces():
            if (len(x) > a):
                a = len(x)
        return a - 1

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



def SCG(G):
    '''
    A simplicial complex is associated with a graph by means of its maximal cliques.
    '''
    v = []
    for x in list(nx.find_cliques(G)):
        w = []
        for y in x:
            w.append(y)
        v.append(w)
    return SCMaximalFacets(v)



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

def scdiff(s2, s1):
    """
    Returns a list with the difference of the simplicial complexes, sorted by dimension.
    """
    v2 = s2.faces()
    v1 = s1.faces()
    v3 = list(set(v2)-set(v1))
    return sorted(v3, key=len)

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
    osv = []
    n = len(sf)
    sc = sf[n-1]
    dv = []
    for i in sf[0].faces():
        if (i != ()):
            sv.append(i)
            osv.append([i])
    dv.append(0)
    dv.append(len(sv))
    for i in range(n-1):
        for j in scdiff(sf[i+1],sf[i]):
            sv.append(j)
            osv.append([j])
        dv.append(len(sv))
    nf = len(sv)
    M = csr_matrix((nf, nf), dtype=np.int8).toarray()
    for i in range(nf):
        if (len(sv[i]) > 1):
            for j in sc.boundary()[sv[i]]:
                M[sv.index(j), i] = 1
    for i in range(nf):
        if (np.count_nonzero(M[:,i]) > 0):
            nz = np.max(np.nonzero(M[:,i]))
            fo = np.min(np.nonzero(M[nz,:]))
            while (fo < i):
                M[:,i] = M[:,i] + M[:,fo]
                osv[i].append(sv[fo])
                for j in range(nf):
                    M[j,i] = M[j,i] % 2
                if (np.count_nonzero(M[:,i]) > 0):
                    nz = np.max(np.nonzero(M[:,i]))
                    fo = np.min(np.nonzero(M[nz,:]))
                else:
                    fo = i + 1
    pi = {}
    for i in range(nf):
        if (np.count_nonzero(M[:,i]) == 0):
            fo = np.min(np.nonzero(M[i,:]))
            fz = np.max(np.nonzero(M[:,fo]))
            if (i < fz):
                av1 = fin_d(i, dv)
                av2 = dv[len(dv) - 1]
                if (av2-av1 > 0):
                    pi[tuple(osv[i])] = [av1, av2]
            else:
                av1 = fin_d(i, dv)
                av2 = fin_d(fo, dv)
                if (av2-av1 > 0):
                    pi[tuple(osv[i])] = [av1, av2]
    return M, pi


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



#sc = SCMaximalFacets([[1,2,3],[1,4,3]])
f1 = [(2,), (1,), ()]
f2 = [(1, 2), (2,), (4,), (2, 3), (1,), (), (3,)]
f3 = [(1, 2), (2,), (4, 3), (1, 4), (4,), (2, 3), (1,), (), (3,)]
f4 = [(1, 2), (2,), (4, 3), (1, 4), (4,), (2, 3), (1,), (), (3,), (1, 3)]
f5 = [(1, 3), (1, 2), (2,), (4, 3), (1, 2, 3), (4,), (1, 4), (2, 3), (1,), (), (3,)]
f6 = [(1, 3), (1, 2), (2,), (4, 3), (1, 2, 3), (4,), (1, 4), (2, 3), (1,), (1, 4, 3), (), (3,)]
sf = [SCset(f1),SCset(f2),SCset(f3),SCset(f4),SCset(f5),SCset(f6)]
M, pi = boundary_matrix(sf)
plot_pd(M, pi)
# print(scdiff(SCset(f1),SCset(f2)))
# print(scdiff(SCset(f2),SCset(f3)))
# print(scdiff(SCset(f3),SCset(f4)))
# print(scdiff(SCset(f4),SCset(f5)))
# print(scdiff(SCset(f5),SCset(f6)))
#sc1 = SCset(f3)
#print(sc1.p_simplices(1))
#print(sc.p_simplices(1))
#print(sc1.boundary())
#print(sc.boundary())
#print(sc.boundary())
# print(eq_elements((3,), (3,)))
# g1 = nx.DiGraph()
# g1.add_nodes_from([1,2])
# g2 = g1.copy()
# g2.add_nodes_from([4])
# g2.add_edges_from([(1,2), (2,3)])
# g3 = g2.copy()
# g3.add_edges_from([(3,4),(4,1)])
# g4 = g3.copy()
# g4.add_edges_from([(1,3)])
# g5 = g4.copy()
# g5.add_edges_from([(3,4),(4,1)])
# nx.draw_networkx(g2)
# plt.show()
# print(sc.faces())
# sc = SCMaximalFacets([[0,1,2]])
# print(sc.s_dimension().keys())
