# Import libraries
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import multiprocessing

def get_neighbors(threshold_df,i):
	'''
	Inputs:
	threshold_df (modified) = contains unique threshold sequences df[0,:], threshold developabilities df[1,:], 
	and threshold embeddings df[2,:]
	These values correspond to the sequences for a given nested sampling step (along the 50 steps)
	that have the lowest corresponding developability in the step of all other sequences with
	desired embeddings of corresponding unique sequences

	i = the current nested sampling step number recorded in loops of 50 within /ns_walkers/

	Outputs: 
	edges = list of tuples of (current nested sampling step i , ); will have either 1 or 2 entries (neighbors);
	these edges will be used to eventually construct a graph of all of the NS runs that will be connected if: 
	1) developabilities are lower than some cutoff, and 
	2) the embeddings for the corresponding NS_run are the closest for 0-2 neighboring "node" NS_runs
	'''
	if i%50==0:
		print(i,'/',len(threshold_df))

    # Instantiate variables
	n_neighbors=2
	A=[]

    # 
	cur_energy = threshold_df.iloc[i]['Energy']
	if cur_energy == min(threshold_df['Energy']):
		return []

    # Get positions from which to build graph
	positions = np.stack(threshold_df[threshold_df['Energy'] < cur_energy]['Positions'])

    # Condition for if there is only one neighbor
	if len(positions) < n_neighbors:
		n_neighbors=1

	# Find closest neighbor
	knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',n_jobs=1,algorithm='brute').fit(positions)
	d, nn = knn.kneighbors(np.stack([threshold_df.iloc[i]['Positions']]),return_distance=True)

	# Connect to all sequences that distance away
	neigh = NearestNeighbors(radius=d[0][-1], metric='euclidean',n_jobs=1,algorithm='brute').fit(seqs_lower)
	A = neigh.radius_neighbors(np.stack([threshold_df.iloc[i]['Positions']]),return_distance=False)[0]
	if len(A)<len(nn[0]):
		A = nn[0]

    # Build edges
	edges = []
	for a in A:
		edges.append((i,a))

    # Return the edges
	return edges

def get_points(self,G,x_min,x_max,frac_of_seq,point_list):
	'''
	Inputs:
	G = the (hopefully 1 single fully connected graph) of the NS run from make_graph(): every node is a NS step;
	each node has 1-2 edges that connect to NS step nodes for which embeddings were closest
	and the corresponding developabilities were at or below the original node's treshold developability

	x_min and x_max = left to right location of the disconnectivity plot
	frac_of_seq = help for when the graph splits; this is set to 1 initially, then sets %config space for bounds for subgraphs
	point_list = empty list to start; then running list;  will populate with [[x, Emax],...] for every step in the ns_run in a 
	way amenable to plotting; 
		
	Note: Emax=(lowest)threshold dev for the ns_step within ns_run

	Output: 
    point_list (modified) = running list of [[x, Emax],...] for every step in the ns_run in a way amenable to plotting
    '''

    # Collect the number of subgraphs
    n_subgraphs=len(list(G.subgraph(c) for c in nx.connected_components(G))) 

    
    while n_subgraphs==1:
        E_max = self.threshold_df.iloc[max(G.nodes)]['Energy']#maximum energy=penalty=minimum developability
        width = sum(self.dos_df[self.dos_df['Energy']>=E_max]['DoS'])#width of config space depends on all dos that have energies worse than emax
        width_adj = (width*frac_of_seq)/2#this "helps for when the graph splits" using frac_of_seq=1 normally
        
        #find and record bounds in config space for the Emax (lowest developability)
        x_center = np.average([x_min,x_max])
        x_min = max(x_center-width_adj,x_min)#don't want to go below 0
        x_max = min(x_center+width_adj,x_max)#don't want to go above 1
        point_list.append([x_min, E_max])
        point_list.append([x_max, E_max])
        
        #we've recorded the lhs and rhs corresponding to emax, so now find, record, and 
        #remove all ns_run step nodes that have emax labels (the lowest threshold developability)
        nodes_to_remove = np.where(self.threshold_df['Energy']==E_max)[0]
        for n in nodes_to_remove:
            if n in G.nodes:
                G.remove_node(n)

        #now that we've removed nodes that we've recorded, check if we caused a split in the graph; record #subgraphs formed (if any)
        n_subgraphs=len(list(G.subgraph(c) for c in nx.connected_components(G))) 

        # if (x_max-x_min)<0.1 or len(G.nodes)<2:

        #if we only have 1 node left, we need to find/record this optimal point at the "peak" of the landscape as the center point 
        #along with the correspdonding developability which is determined here via max(remaining_nodes)
        if len(G.nodes)==1:
                point_list.append([x_center,self.threshold_df.iloc[max(G.nodes)]['Energy']])
                del(G)
                return point_list

    #now we've most likely broken 1 large graph into 2 or more subgraph (phases) and (of course) have more than 1 node total to examine
    if n_subgraphs>1:
        G_list=list(G.subgraph(c) for c in nx.connected_components(G))
        g_size_list = []
        
        #record number nodes in each of the subgraphs
        for g in G_list:
            g_size_list.append(len(g.nodes))


        g_frac_list = np.array(g_size_list)/sum(g_size_list)#find how much remaining room we should give to each subgraph based on the #nodes in each subgraph compared to total
        g_width = x_max-x_min#the remaining room to place nodes depends on the starting lhs/rhs of the first full graph

        g_idx = np.argsort(g_frac_list) #rank the relative graph #nodes' INDICES for each graph small to large
        x_min_g = x_min
        new_points_list=point_list
        
        #go over all graph fractions' INDICES, starting from small to large fractions
        for gidx in g_idx: 
            g = G_list[gidx] #find the current "smallest" graph out
            g_frac = g_frac_list[gidx] #find the corresponding smallest graph's fraction
            x_max_g = x_min_g + g_width*g_frac #find current available width of config space on rhs
            
            #if the rhs width of the subgroup's config space is beyond the rhs width of the original graph, set it to that of the original (full) graph
            if x_max_g > x_max: 
                x_max_g = x_max
            # if (x_max_g-x_min_g)>0.1 and len(g.nodes)>1:

            #now that we've resized bounds of the graph we'll draw, do a recursion on the subgraph we just redrew the bounds for:
            #now the frac_of_seq is "frac_of_seq (1 originally) * g_frac" in which g_frac is the fraction of config space the 
            #subgraph should take up (no long 1 since it's not the full graph)
            p=self.get_points(g.copy(),x_min_g,x_max_g,frac_of_seq*g_frac,[])

            #0 is false; 1 is true; checking if the list exists at all: are there any points associated with that subgraph we just examined?
            #record all points that we found in the get_points() call
            if p:
                for pt in p:
                    new_points_list.append(pt)

            #this most likely shifts over the bounds for the next subgraph to be examined: 
            #the rhs of the current subgraph will be the lhs/lower bound on the proceeding subgraph
            x_min_g = x_max_g

        #update the current point list and delete the current list of graphs; we will consult the if statement above to recreate the 
        #remaining subgraphs and assign them to a new G_list
        point_list=new_points_list
        del(G_list)

    #most likely for safety: remove the current graph
    del(G)
    return point_list

###############################
# DISCONNECTIVITY GRAPH CLASS #
###############################
class disconnectivity_graph():
    '''
    Disconnectivity graph class
    ___________________________
    This class will set up and plot a disconnectivity graph from a nested sampling output.
    
    ATTRIBUTES:
    ===========
    threshold_df : dataframe of threshold energies and corresponding positions
    dos : density of states list 
    graph : connected graph of points
    
    ___________________________
    '''

    def __init__(self, energy_file, nlive):
        '''
        Initialize disconnectivity graph object. This algorithm will recursively build up
        the disconnectivity graph

        @param energy_file : the file containing the threshold energies and positions of each particle
        @param nlive : the number of live points
        '''

        # Load in data
        self.threshold_df = pd.read_pickle(energy_file, headers=['Position', 'Emax', 'Random'])

        # Preprocess
        positions = np.stack(self.threshold_df['Position'])
        _, unique_idx = np.unique(positions, return_index=True, axis=0)
        self.threshold_df = self.threshold_df[unique_index].sort_index()

        # Get DOS - Assume geometric compression of phase space
        self.dos = (1 / (nlive + 1)) * nlive / ((nlive + 1) ** np.arange(1, len(self.threshold_df)+1)) 

        # Create graph
        G = nx.graph()
        G.add_nodes_from(list(range(len(self.threshold_df))))
        get_neighbors_filled = partial(get_neighbors, self.threshold_df.copy())

        # Parallelize
        pool = multiprocessing.Pool(processes=15)
        i = list(range(len(self.threshold_df)))
        (edge_list) = pool.map(get_neighbors_filled, i)
        pool.close()

        # Assign edges
        for edges in edge_list:
            G.add_edges_from(edges)

        # Save graph
        self.graph = G

        # Get point list
        point_list = []
        point_list = get_points(self.graph.copy(), 0, 1, 1, point_list)
        point_list = np.array(point_list)
        self.point_list = point_list[np.argsort(point_list[:,0])]

    def make_fig(self):

        # Instantiate figure
        fig, ax = plt.subplots(1,1,figsize=[2, 2],dpi=1200)
		ax.plot(point_list[:,0],point_list[:,1],linewidth=0.5,color='black',marker='.',markersize=0.5)
		ax.tick_params(labelsize=6,which='both',colors='black')
		ax.set_ylim(bottom=-1.5,top=2.5)    #trying to standardize the axes bound
		ax2=ax.secondary_yaxis('right',functions=(self.fwd,self.rev))
		ax2.tick_params(labelsize=6,which='both',colors='black')
		ax2_tick_list=[]
		ax2_tick_lables=[]
        ax.set_xticks([])
		ax.set_xlabel('Configuration Space',fontsize=6)
		ax.set_ylabel('Energy',fontsize=6)
		ax2.set_ylabel('Fraction of Space With Higher Energy',fontsize=6,color='black')

        # Save image of figure
		fig.tight_layout()
		image_name='./discont/'+NS_run.savename+'_discont.png'
		fig.savefig(image_name)