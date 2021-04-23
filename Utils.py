####### Package imports
import numpy as np
import networkx as nx

##### Method to correctly read out the index of a tuple in a list
##### ignoring the order of the elements
def redefined_index(list_of_tuples,element):
    """redefine implemented index method for list 
    
    Parameters
    ----------
    list_of_tuples : list
        A list containing tuples
    element : tuple
        A single tuple whose position in the list is calculated ignoring tuple orientation
    
    Returns
    -------
    integer 
        index of element in list_of_tuples ignoring orientation of tuples
    """
    
    assert isinstance(list_of_tuples,list)
    assert isinstance(element,tuple)
    
    try:
        index = list_of_tuples.index(element)
    except ValueError:
        index = list_of_tuples.index(element[::-1])
    return index

########## Calculate PTDF matrix
def PTDF_matrix(G,I=np.array([])):
    """Calculate Power Transfer Distribution Factor (PTDF) Matrix for power injections along graph's edges
    NOTE: This PTDF matrix is already multiplied from the right by the graphs 
    incidence matrix, thus assuming power injections only to take place at the terminal
    ends of edges and resulting in a nof_edges x nof_edges matrix
    
    Parameters
    ----------      
    G : networkx graph
         Graph based on which Laplacian matrix and thus PTDF matrix is calulated
    I : (optional) numpy array
         Represents oriented edge node incidence matrix 
    
    
    Returns
    -------
    numpy array
         Power Transfer Distribution Factor matrix of dimension number_of_edges x number_of_edges
    """
    
    B = nx.laplacian_matrix(G).A
    if not I.size:
        I = nx.incidence_matrix(G,oriented=True).A

    edges = [(list(G.nodes()).index(u),list(G.nodes()).index(v)) for u,v in G.edges()]
    line_weights = np.array([B[e] for e in edges])
    B_d = -np.diag(line_weights)
    #multi_dot is supposed to find the most efficient way of performing the matrix multiplication
    #latter term is B inverse matrix
    try: 
        #implicit matrix inversion
        B_inv = np.linalg.solve(B, I)
        PTDF_matrix = np.linalg.multi_dot([B_d,I.T,B_inv])
        ### This sometimes results in Singular Matrix error
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B)
        PTDF_matrix = np.linalg.multi_dot([B_d,I.T,B_inv,I])
    return PTDF_matrix

####
def shortest_edge_distance(G,e1,e2,weighted=False):
    """Calculate the edge distance between edge e1 and edge e2 by taking the minimum of all
    shortest paths between the nodes
    
    Parameters
    ----------      
    G : weighted or unweighted networkx graph
         Graph in which distance is to be calculated
    e1, e2 : tuples 
         Represent edges between which distance is to be calculated
    weighted : boolean
         If True, edge distance is calculated based on inverse edge weights
    
    Returns
    -------
    float 
         The length of the shortest path between e1 and e2 in G
    """

    assert isinstance(G,nx.Graph)
    assert isinstance(e1,tuple)
    assert isinstance(e2,tuple)
    
    possible_path_lengths = []
    F = G.copy()
    weight = nx.get_edge_attributes(F,'weight')
    if ((not len(weight)) or (not weighted)):
        weight = {e:1.0 for e in F.edges()}
        nx.set_edge_attributes(F,weight,'weight')
    inv_weight = {}
    for key in weight.keys():
        inv_weight[key] = 1/weight[key]
    nx.set_edge_attributes(F,inv_weight,'inv_weight')
   
    for i in range(2):
        for j in range(2):
            possible_path_lengths.append(nx.shortest_path_length(F,source=e1[i],target=e2[j],weight='inv_weight'))
    path_length = min(possible_path_lengths)+(F[e1[0]][e1[1]]['inv_weight']+F[e2[0]][e2[1]]['inv_weight'])/2
    return path_length   

##### Synthetic graphs model 
def create_synthetic_ER_graphs(N1,N2,p1,p2,c1,c2,mu):
    """Create two Erdös Renyi random graphs that are connected to
    each other at c1*N1 and c2*N2 vertices with probability mu
    
    Parameters
    ----------      
    N1, N2 : integers
        Parameter for number of nodes in ER graphs F and G
    p1, p2 : floats
        Parameters for connections probability in ER graphs
    c1, c2 : floats
        Parameter representing the share of nodes in F and G that potentially connect to each other
    mu : float
        connection probability for connections between F and G at nodes chosen by c1 and c2
    
    Returns
    -------
    networkx graph
        Result of the random connection of the two ER subgraphs with probability mu. 
        Graph object contains two attributes accesible via G.graph['attribute']:
        'connectors' contains a dictionary that, for each node in the graph, indicates
                     whether the node is among the cN1 and cN2 nodes chosen 
                     as potentially connecting (value 1) or not (value 0)
        'subgraph_nodes' contains a dictionary that for each node n indicates whether 
                         it belongs to the first subgraph (value 1) or 
                         second subgraph (value 0)
    """

    #### Generate two ER random graphs
    G = nx.gnp_random_graph(n=N1,p=p1)
    F = nx.gnp_random_graph(n=N2,p=p1)
    
    #### Nodes of F and G are both named 1... N initally, so rename nodes of F in order to 
    #### compose the graphs to a single large graph H
    mapping = {n:n+N1 for n in F.nodes()}
    F = nx.relabel_nodes(F,mapping=mapping)
    H = nx.compose(F,G)
    
    #### Now randomly choose int(c*N) nodes from both graphs that will connect to the other graph
    connectors_G = list(G.nodes())[:int(N1*c1)]
    connectors_F = list(F.nodes())[:int(N2*c2)]
    maximum_number_of_connections = len(connectors_G)*len(connectors_F)
    
    #### save connectors in node attributes 
    nc = {n:0.0 for n in H.nodes()}
    for n in connectors_G:
        nc[n] = 1.0
    for n in connectors_F:
        nc[n] = 1.0
    nx.set_node_attributes(H,nc,'connectors')
    
    #### Save information about nodes belonging to subgraph G
    nodes_G = {n:0.0 for n in H.nodes()}
    for n in G.nodes():
        nodes_G[n] = 1.0
    nx.set_node_attributes(H,nodes_G,'subgraph_nodes')
    
    #### Set up variables to count the share of connections, in order to be able to compare it to mu
    current_number_of_connections = 0
    fraction_of_connections = 0
    
    ### now randomly add edges until required value of mu is reached
    while fraction_of_connections<mu:
        node_G = connectors_G[np.random.randint(low=0,high=len(connectors_G))]
        node_F = connectors_F[np.random.randint(low=0,high=len(connectors_F))]
        if not H.has_edge(node_G,node_F):
            H.add_edge(node_G,node_F)
            current_number_of_connections +=1
            fraction_of_connections = current_number_of_connections/maximum_number_of_connections
    return H


#### Calculate flow ratio R(d,l)
def calc_flow_ratio(H,F,G):
    """Calculate flow ratio for all possible trigger links and distances in Graoh H
    from links in subgraph F of H to subgraph G in H and vice versa
    
    Parameters
    ----------      
    H : Networkx graph
       representing the whole graph
    G : Networkx graph
       subgraph of H representing one of the two modules of H
    F : Networkx graph
       subgraph of H representing the other module of H
    
    Returns
    -------
    numpy array
          contains distances [0] and flow ratios [1] evaluated
          for all possible trigger links
    """
    
    assert isinstance(H,nx.Graph)
    assert isinstance(F,nx.Graph)
    assert isinstance(G,nx.Graph)
    
    
    ### get the indices of all edges in F and in G
    edges_in_G_indices = [redefined_index(list_of_tuples=list(H.edges()),element=e) for e in G.edges()]
    edges_in_F_indices = [redefined_index(list_of_tuples=list(H.edges()),element=e) for e in F.edges()]
    
    ### calculate PTDF matrix for the given graph
    PTDF = PTDF_matrix(H)
    flow_ratio = []
    distances = []
    ### Now iterate over all possible trigger links in G
    for trigger_index in edges_in_G_indices:
        trigger_link = list(H.edges())[trigger_index]

        ### calculate distance from trigger link to all possible other links
        edge_distance_uw = [shortest_edge_distance(G = H,e1 = e,e2 = trigger_link,weighted = False) for e in H.edges()]
        max_dist = np.max(edge_distance_uw)
        for d in np.arange(1,max_dist):
            ### get all edges at a distance d to the trigger link
            edges_distance_d_G = [i for i in edges_in_G_indices if edge_distance_uw[i]==d]
            edges_distance_d_F = [i for i in edges_in_F_indices if edge_distance_uw[i]==d]
            ### calculate the absolute ration of mean flows at this distance, if there are links in G and F at the given distance
            if len(edges_distance_d_G) and len(edges_distance_d_F):
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12))
                distances.append(d)
    ### Iterate over all possible trigger links in F
    for trigger_index in edges_in_F_indices:
        trigger_link = list(H.edges())[trigger_index]
        
        ### calculate distance from trigger link to all possible other links
        edge_distance_uw = [shortest_edge_distance(G = H,e1 = e,e2 = trigger_link,weighted = False) for e in H.edges()]
        max_dist = np.max(edge_distance_uw)
        for d in np.arange(1,max_dist):
            ### get all edges at a distance d to the trigger link
            edges_distance_d_G = [i for i in edges_in_G_indices if edge_distance_uw[i]==d]
            edges_distance_d_F = [i for i in edges_in_F_indices if edge_distance_uw[i]==d]
            ### calculate the absolute ration of mean flows at this distance, if there are links in G and F at the given distance
            if len(edges_distance_d_G) and len(edges_distance_d_F):
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12))
                distances.append(d)
    distances = np.array(distances)
    flow_ratio = np.array(flow_ratio)
    ratio_and_distances = np.array([distances,flow_ratio])
    return ratio_and_distances

def calc_flow_ratio_single_link(H,F,G,trigger_link):
    """Calculate flow ratio for a single trigger link
    
    Parameters
    ----------      
    H : Networkx graph
       representing the whole graph
    G : Networkx graph
       subgraph of H representing one of the two modules of H
    F : Networkx graph
       subgraph of H representing the other module of H
    trigger_link : tuple
       represents the edge that fails in order to calculate flow ratio
    
    Returns
    -------
    numpy array
         contains distances [0] and flow ratios [1] evaluated
         for all possible trigger links
    """
    
    assert isinstance(H,nx.Graph)
    assert isinstance(F,nx.Graph)
    assert isinstance(G,nx.Graph)
    assert isinstance(trigger_link,tuple)
        
    ### Check if link is located in module G or module F
    trigger_module = ''
    if G.has_edge(*trigger_link):
        trigger_module = 'G'
    elif F.has_edge(*trigger_link):
        trigger_module = 'F'
    else:
        print('Error, edge ' + str(trigger_link) + 'neither located in subgraph G, nor in subgraph F')
        return 
        
    
    ### get the indices of all edges in F and in G
    edges_in_G_indices = [redefined_index(list_of_tuples = list(H.edges()),element = e) for e in G.edges()]
    edges_in_F_indices = [redefined_index(list_of_tuples = list(H.edges()),element = e) for e in F.edges()]
    
    ### calculate PTDF matrix for the given graph
    PTDF = PTDF_matrix(H)
    

    ### calculate distance from trigger link to all possible other links
    edge_distance_uw = [shortest_edge_distance(G = H,e1 = e,e2 = trigger_link,weighted = False) for e in H.edges()]
    max_dist = np.max(edge_distance_uw)
    
    flow_ratio = []
    distances = []
   
    trigger_index = redefined_index(list(H.edges()),trigger_link)

    for d in np.arange(1,max_dist):
        ### get all edges at a distance d to the trigger link
        edges_distance_d_G = [i for i in edges_in_G_indices if edge_distance_uw[i]==d]
        edges_distance_d_F = [i for i in edges_in_F_indices if edge_distance_uw[i]==d]
        ### calculate the absolute ratio of mean flows at this distance, if there are links in G and F at the given distance
        if len(edges_distance_d_G) and len(edges_distance_d_F):
            if trigger_module == 'G':
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12))
            elif trigger_module == 'F':
                flow_ratio.append(np.round(np.mean(np.abs(PTDF[edges_distance_d_G,trigger_index])),decimals = 12)/np.round(np.mean(np.abs(PTDF[edges_distance_d_F,trigger_index])),decimals = 12))
            distances.append(d)
            
    distances = np.array(distances)
    flow_ratio = np.array(flow_ratio)
    ratio_and_distances = np.array([distances,flow_ratio])
    return ratio_and_distances
    
   
#############################################################################################################################################
############################# Following functions are used to analyse perturbations of networks isolators
def calc_min_norm(vs):
    """Calculate minimum norm between rows of numpy array vs

    Parameters
    ----------   
    vs : numpy array
      of which mutual row products are calculated
    
    Returns
    -------
    float
         minimum, normalized row norm of array vs
    """
    
    assert isinstance(vs,np.ndarray)
    norms = []
    for i in range(vs.shape[0]):
        for j in range(i+1,vs.shape[0]):
            norms.append(np.dot(vs[i],vs[j])/(np.sqrt(np.sum(vs[i]**2)*np.sum(vs[j]**2))))
    return np.min(norms)

def perturb_isolator(H,alpha,base_vectors):
    """Perturb the isolator with perturbation strength alpha
    
    Parameters
    ----------
    H :           weighted networkx graph
           Graph to be analysed containing isolator with weighted edges
    alpha :       float
           Represents the perturbation strength
    base_vectors: List of lists of tuples
           Shape is nof_rows x nof_columns of the adjacency matrix representing the isolator motif,
           but contains the edges of the isolator instead
    
    Returns
    -------
    weighted networkx graph 
           Graph with weights modified after perturbation 
    float
           Coherence statistics of the weight matrix representing the isolator
           being zero for a perfect isolator and assuming values up to one
    """
    weight_dict = nx.get_edge_attributes(H,'weight')
    
    basis_weights = np.zeros((len(base_vectors),len(base_vectors[0])))
    
    for i in range(basis_weights.shape[0]):
        for j in range(basis_weights.shape[1]):
            ### read out edge in the proper orientation
            edge = list(H.edges())[redefined_index(list(H.edges()),base_vectors[i][j])]
            basis_weights[i,j] = weight_dict[edge]
    
    ## randomly choose a vector to be perturbed
    r = np.random.randint(low=0,high=len(base_vectors))
    ## contains edges as tuples of the vector to be perturbed
    perturbed_vector = [list(H.edges())[redefined_index(list(H.edges()),edge)] for edge in base_vectors[r]]
    ## get edge weights and norm of perturbed vector
    perturbed_weights = np.array([weight_dict[edge] for edge in perturbed_vector])
    norm = np.sqrt(np.sum(np.array(perturbed_weights)**2))
    
    ## create perturbation vector from a uniform distribution
    perturbation = np.random.uniform(low=-1.,high=1.,size=len(perturbed_vector))
    
    perturbed_weights += alpha*perturbation*norm
    
    ## Edge weights should stay positive, which they will nevertheless, if perturbation parameter is weak
    ## Now update weights
    for j in range(len(perturbed_vector)):
        weight_dict[perturbed_vector[j]] = np.abs(perturbed_weights[j])
        weight_dict[perturbed_vector[j][::-1]] =  np.abs(perturbed_weights[j])
        basis_weights[r,j] = np.abs(perturbed_weights[j])
    nx.set_edge_attributes(H,weight_dict,'weight')
    
    coherence_statistics = 1 - calc_min_norm(basis_weights)
    return H, coherence_statistics
