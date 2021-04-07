##################################################################################################
##################################################################################################
######################## Create weakly coupled ER random graphs ##################################
##################################################################################################

import networkx as nx
import numpy as np

# Local import
import Utils

##### Model parameters for creation of ER random graphs and their mutual connections
# We assume both subgraphs to have N1 = N2 = N nodes and equal connection probability 
# p. We also assume the share of connections between the two subgraphs to be equal
# c1 = c2 = c

N1 = 30
N2 = N1
p1 = 0.3
p2 = p1
c1 = 0.1
c2 = c1
mu = 0.2

H = Utils.create_synthetic_ER_graphs(N1,N2,p1,p2,c1,c2,mu)

##################################################################################################
######################## Calculate flow ratio for graph ##########################################
##################################################################################################


### extract the two subgraphs from H to calculate the flow ratio
subgraph_nodes = nx.get_node_attributes(H,'subgraph_nodes')

nodes_G = [n for n in H.nodes() if subgraph_nodes[n]]
nodes_F = [n for n in H.nodes() if not subgraph_nodes[n]]
G = H.subgraph(nodes_G)
F = H.subgraph(nodes_F)

### extract the nodes that are potentially connecting
### from both graphs F and G
connectors = nx.get_node_attributes(H,'connectors')
connectors_G = [n for n in G.nodes() if connectors[n]]
connectors_F = [n for n in F.nodes() if connectors[n]]

##### Now calculate ratio of flows for increasing connectivity as encoded by mu

maximum_number_of_connections = len(connectors_F)*len(connectors_G)
current_number_of_connections = len(H.edges())-len(G.edges())-len(F.edges())
fraction_of_connections = current_number_of_connections/maximum_number_of_connections

H0 = H.copy()

mus = np.arange(fraction_of_connections,1.0,1/maximum_number_of_connections)

flow_ratios = np.zeros((len(mus),3))
count = 0

between_edges = []
for mu in mus:
    ### now randomly add edges until required value of mu is reached
    while fraction_of_connections<mu:
        node_G = connectors_G[np.random.randint(low=0,high=len(connectors_G))]
        node_F = connectors_F[np.random.randint(low=0,high=len(connectors_F))]
        if not H0.has_edge(node_G,node_F):
            H0.add_edge(node_G,node_F)
            current_number_of_connections +=1
            fraction_of_connections = current_number_of_connections/maximum_number_of_connections
            between_edges.append((node_G,node_F))
    flow_ratio = Utils.calc_flow_ratio(H,F,G)
    print(flow_ratio)
    #### average over all flow ratios
    #### entry 0 contains distances, entry 1 flow ratios
    flow_ratios[count,:] = np.array([np.quantile(flow_ratio[1],q=0.5),
                                      np.quantile(flow_ratio[1],q=0.25),
                                      np.quantile(flow_ratio[1],q=0.75)])
                                      

###############################################################################################
################### Test robustness of network isolator motif #################################
###############################################################################################

### Model parameters
nof_perturbs = 1000
### this is the strength of perturbation applied at each iteration
alpha = 0.05 

##### We demonstrate the approach for two initially unconnected ER random graphs
##### which we connect through a network islator

###############################################################################################
############################# Graph creation

##### Model parameters for creation of ER random graphs
N = 20
p = 0.3

#### Generate two ER random graphs
G = nx.gnp_random_graph(n=N,p=p)
F = nx.gnp_random_graph(n=N,p=p)

#### Nodes of F and G are both named 1... N initally, so rename nodes of F in order to 
#### compose the graphs to a single large graph H
mapping = {n:n+N for n in F.nodes()}
F = nx.relabel_nodes(F,mapping=mapping)
H = nx.compose(F,G)

################################################################################
############################ Isolator creation

#### Randomly choose nodes from both graphs for isolator
nof_nodes_in_isolator_G = 3
nof_nodes_in_isolator_F = 4

indices_G = np.arange(len(G.nodes()))
np.random.shuffle(indices_G)
indices_F = np.arange(len(F.nodes()))
np.random.shuffle(indices_F)

nodes_in_isolator_G = [list(G.nodes())[index] for index in indices_G[:nof_nodes_in_isolator_G]]
nodes_in_isolator_F = [list(F.nodes())[index] for index in indices_F[:nof_nodes_in_isolator_F]]

edges_in_isolator = list(itertools.product(nodes_in_isolator_G,nodes_in_isolator_F))
H.add_edges_from(edges_in_isolator)

#### Base vectors of mutual connections expressed in terms of edges
base_vectors = [list(itertools.product([nodes_in_isolator_G[i]],nodes_in_isolator_F)) for i in range(len(nodes_in_isolator_G))]

#### Add random edge weights to the graph
weights_array = np.random.normal(loc=10.0,scale=1.,size=len(H.edges()))
weights_dict = {}
count = 0
for e in list(H.edges()):
    weights_dict[e] = weights_array[count]
    weights_dict[e[::-1]] = weights_dict[e]
    count += 1

### now create network isolator by making the columns of the connectivity matrix linearly dependent
for i in range(nof_nodes_in_isolator_F):
    for j in range(1,nof_nodes_in_isolator_G):
        e = (nodes_in_isolator_G[j],nodes_in_isolator_F[i])
        weights_dict[e] = weights_dict[(nodes_in_isolator_G[0],nodes_in_isolator_F[i])]#basis_weights[0,i]
        weights_dict[e[::-1]] = weights_dict[e]
nx.set_edge_attributes(H,weights_dict,'weight')    

###############################################################################################
########################### Run perturbation algorithm
    
coherence_statistics = np.zeros(nof_perturbs)
flow_ratios = np.zeros((nof_perturbs,3))

for index in range(nof_perturbs):
    
    H,coherence_statistics[index] = Utils.perturb_isolator(H,alpha,base_vectors)
    print(coherence_statistics[index])
    
    flow_ratio = Utils.calc_flow_ratio(H,F,G)
    ### Caculate median, 25 quantile and 75 quantile
    flow_ratios[index,:] = np.array([np.quantile(flow_ratio[1],q=0.5),
                                      np.quantile(flow_ratio[1],q=0.25),
                                      np.quantile(flow_ratio[1],q=0.75)])

