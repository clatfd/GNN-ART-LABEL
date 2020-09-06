import numpy as np
import networkx as nx
import copy

BOITYPENUM = 23
VESTYPENUM = 25

#node matching edges
edgefromnode = [[] for i in range(BOITYPENUM)]
#node 0 can have any edge type
edgefromnode[0] = [0]
edgefromnode[1] = [1]
edgefromnode[2] = [2]
edgefromnode[3] = [1,3,7]
edgefromnode[4] = [2,4,8]
edgefromnode[5] = [7,9,11]
edgefromnode[6] = [8,10,11]
edgefromnode[7] = [3,5,5]
edgefromnode[8] = [4,6,6]
edgefromnode[9] = [1,1,23]
edgefromnode[10] = [2,2,24]
edgefromnode[11] = [23]
edgefromnode[12] = [24]
#skip m23
edgefromnode[15] = [14]
edgefromnode[16] = [15]
edgefromnode[17] = [14,15,16]
edgefromnode[18] = [16,17,18]
edgefromnode[19] = [17,19,21]
edgefromnode[20] = [18,20,22]
edgefromnode[21] = [1,1,21]
edgefromnode[22] = [2,2,22]

nodeconnection = [[] for i in range(BOITYPENUM)]
nodeconnection[3] = [1,5,7]
nodeconnection[4] = [2,6,8]
nodeconnection[18] = [17,19,20]


#def getvesname(id):
VesselName = [None for i in range(VESTYPENUM)]
VesselName[1] = "ICA_L";
VesselName[2] = "ICA_R";
VesselName[3] = "M1_L";
VesselName[4] = "M1_R";
VesselName[5] = "M2_L";
VesselName[6] = "M2_R";
VesselName[7] = "A1_L";
VesselName[8] = "A1_R";
VesselName[9] = "A2_L";
VesselName[10] = "A2_R";
VesselName[11] = "AComm";
VesselName[12] = "M3_L";
VesselName[13] = "M3_R";
VesselName[14] = "VA_L";
VesselName[15] = "VA_R";
VesselName[16] = "BA";
VesselName[17] = "P1_L";
VesselName[18] = "P1_R";
VesselName[19] = "P2_L";
VesselName[20] = "P2_R";
VesselName[21] = "PComm_L";
VesselName[22] = "PComm_R";
VesselName[23] = "OA_L";
VesselName[24] = "OA_R";
#    return VesselName[id]

NodeName = [None for i in range(BOITYPENUM)]
NodeName[0] = "Undefined";
NodeName[1] = "ICARoot_L";
NodeName[2] = "ICARoot_R";
NodeName[3] = "ICA/MCA/ACA_L";
NodeName[4] = "ICA/MCA/ACA_R";
NodeName[5] = "A1/A2_L";
NodeName[6] = "A1/A2_R";
NodeName[7] = "M1/M2_L";
NodeName[8] = "M1/M2_R";
NodeName[9] = "OA/ICA_L";
NodeName[10] = "OA/ICA_R";
NodeName[11] = "OA_L";
NodeName[12] = "OA_R";
NodeName[13] = "M2/M3_L";
NodeName[14] = "M2/M3_R";
NodeName[15] = "VARoot_L";
NodeName[16] = "VARoot_R";
NodeName[17] = "BA/VA";
NodeName[18] = "PCA/BA";
NodeName[19] = "P1/P2/Pcomm_L";
NodeName[20] = "P1/P2/Pcomm_R";
NodeName[21] = "Pcomm/ICA_L";
NodeName[22] = "Pcomm/ICA_R";

def matchvestype(starttype,endtype):
    EndCondition = np.zeros((100,100),dtype=np.int8)
    EndCondition[1][3] = 1;
    EndCondition[2][4] = 2;
    EndCondition[3][1] = 1;
    EndCondition[3][5] = 7;
    EndCondition[3][7] = 3;
    EndCondition[4][2] = 2;
    EndCondition[4][8] = 4;
    EndCondition[4][6] = 8;
    EndCondition[5][6] = 11;
    EndCondition[5][3] = 7;
    EndCondition[5][23] = 9;
    EndCondition[5][99] = 9;
    EndCondition[6][4] = 8;
    EndCondition[6][5] = 11;
    EndCondition[6][24] = 10;
    EndCondition[6][99] = 10;
    EndCondition[7][3] = 3;
    EndCondition[7][13] = 5;
    EndCondition[7][29] = 5;
    EndCondition[7][99] = 5;
    EndCondition[8][4] = 4;
    EndCondition[8][14] = 6;
    EndCondition[8][30] = 6;
    EndCondition[8][99] = 6;
    EndCondition[9][11] = 23;
    EndCondition[10][12] = 24;
    EndCondition[11][9] = 23;
    EndCondition[12][10] = 24;
    EndCondition[13][7] = 5;
    EndCondition[13][99] = 12;
    EndCondition[13][25] = 12;
    EndCondition[13][29] = 5;
    EndCondition[14][8] = 6;
    EndCondition[14][99] = 13;
    EndCondition[14][26] = 13;
    EndCondition[14][30] = 6;
    EndCondition[15][17] = 14;
    EndCondition[16][17] = 15;
    EndCondition[17][15] = 14;
    EndCondition[17][16] = 15;
    EndCondition[17][18] = 16;
    EndCondition[18][17] = 16;
    EndCondition[18][20] = 18;
    EndCondition[18][19] = 17;
    EndCondition[19][18] = 17;
    EndCondition[19][21] = 21;
    EndCondition[19][99] = 19;
    EndCondition[19][27] = 19;
    EndCondition[20][22] = 22;
    EndCondition[20][18] = 18;
    EndCondition[20][99] = 20;
    EndCondition[20][28] = 20;
    EndCondition[21][19] = 21;
    EndCondition[22][20] = 22;
    EndCondition[23][99] = 9;
    EndCondition[23][5] = 9;
    EndCondition[23][23] = 9;
    EndCondition[24][99] = 10;
    EndCondition[24][6] = 10;
    EndCondition[24][24] = 10;
    EndCondition[25][99] = 12;
    EndCondition[25][13] = 12;
    EndCondition[25][25] = 12;
    EndCondition[26][99] = 13;
    EndCondition[26][14] = 13;
    EndCondition[26][26] = 13;
    EndCondition[27][99] = 19;
    EndCondition[27][19] = 19;
    EndCondition[27][27] = 19;
    EndCondition[28][99] = 20;
    EndCondition[28][20] = 20;
    EndCondition[28][28] = 20;
    EndCondition[29][7] = 5;
    EndCondition[29][13] = 5;
    EndCondition[29][29] = 5;
    EndCondition[29][99] = 5;
    EndCondition[30][8] = 6;
    EndCondition[30][14] = 6;
    EndCondition[30][30] = 6;
    EndCondition[30][99] = 6;
    return EndCondition[starttype][endtype]

def findedgeid(graph,startnodeid,endnodeid):
    for i,edgei in enumerate(graph.edges()):
        if edgei == (startnodeid,endnodeid) or \
            edgei == (endnodeid,startnodeid):
            return i
    return -1

def nodedist(G,startnodeid,endnodeid):
    cdist = 0
    sp = nx.shortest_path(G,startnodeid,endnodeid)
    for spi in range(1,len(sp)):
        edgei = findedgeid(G,sp[spi-1],sp[spi])
        if edgei!=-1:
            cdist += G.edges()[sp[spi-1],sp[spi]]['dist']
        else:
            print('cannot find id for edge',sp[spi-1],sp[spi])
    return cdist 

#nodeid neighbors are one deg1 and two deg 3+
def neideg3(graph,nodeid):
    neiids = [i[1] for i in list(graph.edges(nodeid))]
    if len(neiids)==1:
        return False
    nndeg = [graph.nodes[ni]['deg'] for ni in neiids]
    print('nndeg',nndeg)
    #in case more than 3 neis, choose larger 3
    if sorted(nndeg)[-3:]==[1,3,3] or sorted(nndeg)==[3,3,3]:
        if sorted(nndeg)[-3:]==[1,3,3]:
            deg_min_node_id = neiids[np.argmin(nndeg)]
            edgeid = findedgeid(graph,nodeid,deg_min_node_id)
            if edgeid!=-1:
                min_branch_dist = graph.edges[list(graph.edges())[edgeid]]['dist']
                if min_branch_dist<10:
                    return False
            else:
                print('no edgeid',nodeid,deg_min_node_id)
        return True
    else:
        return False

def findmaxprob(graph,nodeid,visited,targettype,targetdeg,probnodes,branch_dist_mean,branch_dist_std,vestype=0,majornode=0):
    #first node in visited is the root node
    #nodeid gives the direction to search along that branch
    validnode = {}
    validnode[nodeid] = probnodes[nodeid][targettype]
    pendingnode = [nodeid]
    while len(pendingnode):
        nodestart = pendingnode[0]
        del pendingnode[0]
        neinodes = list(graph.edges(nodestart))
        for ni in neinodes:
            if ni[1] in visited:
                continue
            visited.append(ni[1])
            if targetdeg is None or graph.nodes[ni[1]]['deg']==targetdeg:
                validnode[ni[1]] = probnodes[ni[1]][targettype]
            if graph.nodes[ni[1]]['deg']>=3:
                if vestype in branch_dist_mean.keys():
                    disttoroot = nodedist(graph,ni[1],visited[0])
                    if disttoroot>branch_dist_mean[vestype]+2*branch_dist_std[vestype]:
                        #print('beyond thres',vestype,branch_dist_mean[vestype],branch_dist_std[vestype])
                        continue
                pendingnode.append(ni[1])
    print(validnode)
    if majornode:
        majorvalidnode = copy.copy(validnode)
        for ki in list(majorvalidnode):
            if neideg3(graph,ki)==False:
                del majorvalidnode[ki]
        print('majornode',majorvalidnode)
        if len(majorvalidnode)==0:
            print('major node empty')
            return max(validnode, key=validnode.get)
        else:
            return max(majorvalidnode, key=majorvalidnode.get)
    else:
        return max(validnode, key=validnode.get)

#find id of edges connected to nodeid
def find_node_connection_ids(graph,nodeid):
    edges = list(graph.edges())
    nei_edge_ids = []
    neiedges = graph.edges(nodeid)
    for edgei in neiedges:
        if edgei not in edges:
            edgei = (edgei[1],edgei[0])
        edgeid = edges.index(edgei)
        nei_edge_ids.append(edgeid)
    return nei_edge_ids

def find_nei_nodes(graph,nodeid):
    nei_node_ids = []
    neiedges = graph.edges(nodeid)
    for edgei in neiedges:
        nei_node_ids.append(edgei[1])
    return nei_node_ids
    
def findallnei(graph,nodeid,visited,targetdeg=None):
    validnode = []
    validedge = []
    pendingnode = [nodeid]
    while len(pendingnode):
        nodestart = pendingnode[0]
        del pendingnode[0]
        neinodes = list(graph.edges(nodestart))
        for ni in neinodes:
            if ni[1] in visited:
                continue
            visited.append(ni[1])
            if targetdeg is None or graph.nodes[ni[1]]['deg']==targetdeg:
                validnode.append(ni[1])
                validedge.append(ni)
            if graph.nodes[ni[1]]['deg']>=3:
                pendingnode.append(ni[1])
    return validnode,validedge

def matchbranchtype(nei_node_ids,branch_miss_type,centernodeid):
    maxprobs = 0
    maxli = None
    for li in list(permutations(nei_node_ids, len(nei_node_ids))):
        cprobs = 0
        for item in range(len(nei_node_ids)):
            nodeid = li[item]
            edgeid = findedgeid(graph,centernodeid,nodeid)
            cprob = probnodes[edgeid][branch_miss_type[item]]
            cprobs += cprob
        #print(li,cprobs)
        if cprobs>maxprobs:
            maxprobs = cprobs
            maxli = li
    #print(maxli)
    return maxli

#definition of key sets

def softmax_probs(xi):
    e = np.exp(xi)
    return e/np.sum(e)


