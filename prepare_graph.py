# Functions to generate new graphs from iCafe traces
# Need to have icafe result folder and iCafe-python library

icafefolder = DESKTOPdir+'/iCafe/result'
    
def refreshid(G):
    #replace node id with new id
    Gnew = nx.Graph()
    nodemap = {}
    for newid,node in enumerate(G.nodes(data=True)):
        #print(newid,node[0])
        nodemap[node[0]] = newid
        kwargs = node[1]
        Gnew.add_node(newid,**kwargs)
    for edge in G.edges(data=True):
        #print(edge)
        kwargs = edge[2]
        Gnew.add_edge(nodemap[edge[0]],nodemap[edge[1]],**kwargs)
        
    return Gnew

def generate_graph_icafe(dbname,casename,graphtype='graphsim'):
    REEXP = 0
    picklegraphname = graphfoldername+'/'+graphtype+'/'+dbname+'/'+casename+'.pickle'
    #print(picklegraphname)
    if not os.path.exists(graphfoldername+'/'+graphtype+'/'+dbname):
        os.makedirs(graphfoldername+'/'+graphtype+'/'+dbname)
    if REEXP or not os.path.exists(picklegraphname):
        print('generating graph',dbname,casename)
        if graphtype=='graph':
            cgraph = save_pickle_graph(picklegraphname)
        else:
            cgraph = save_pickle_graph_sim(picklegraphname)
        if cgraph is None:
            return None
    return picklegraphname
        





    
def save_pickle_graph(picklegraphname):
    foldername = picklegraphname[:-1-len(picklegraphname.split('/')[-1])]
    icafem = iCafe(foldername)
    #icafem.loadves()
    #G = icafem.generateGfromves()
    icafem.loadvesnochange()
    icafem.getlandmark()
    G = icafem.generateG(ASSIGNNODE=1,ASSIGNEDGE=1)
    for nodei in G.nodes():
        G.nodes[nodei]['pos'] = G.nodes[nodei]['pos'].pos
        G.nodes[nodei]['deg'] = G.degree[nodei]
        if G.nodes[nodei]['boitype']>22:
            G.nodes[nodei]['boitype'] = 0
    nx.write_gpickle(G, picklegraphname)
    print('Graph saved',picklegraphname)
    return G

def save_pickle_graph_sim(picklegraphname,trim=1):
    dbname = picklegraphname.split('/')[-2]
    casename = picklegraphname.split('/')[-1][:-7]
    foldername = icafefolder+'/'+dbname+'/'+casename
    print(foldername)
    icafem = iCafe(foldername)
    if len(icafem.snakelist) == 0:
        print('no swc snake list')
        return None
    icafem.getlandmark(IGNOREM3=1)
    Gs = icafem.generateSimG(ASSIGNNODE=1,ASSIGNEDGE=1,ASSIGNDIR=1)
    if trim:
        S = []
        for c in nx.connected_components(Gs):
            Gsi = Gs.subgraph(c).copy()
            gsidist = np.sum([Gs.edges[nodei]['dist'] for nodei in Gsi.edges()])
            #print(len(c),gsidist)
            if gsidist>100 and len(Gsi.nodes())>5:
                S.append(Gsi)
            #else:
            #    print('ignore',gsidist)
        print(len(S),'subgraph left')

        #sort based on length
        SSort = []
        for i in np.argsort([len(c) for c in S])[::-1]:
            SSort.append(S[i])
        G = refreshid(nx.compose_all(SSort))
    else:
        G = Gs
        
    VESCOLORS = ['b']+['r']*VESTYPENUM
    NODECOLORS = ['r']+['b']*BOITYPENUM
    posz =  {k: v['pos'].pos[:2] for k, v in G.nodes.items()}
    edgecolors = [VESCOLORS[np.argmax(G.edges[v]['vestype'])] for v in G.edges()]
    nodecolors = [NODECOLORS[G.nodes[n]['boitype']] for n in G.nodes()]
    nx.draw_networkx(G,pos=posz,node_size=30, node_color=nodecolors, edge_color=edgecolors)
    plt.show()

    misslandmk = []
    for i,j in icafem.landmark:
        #print(icafem.VesselName[i],j)
        if j.hashpos() not in icafem.simghash:
            #print(icafem.VesselName[i],'Not Found')
            misslandmk.append(icafem.NodeName[i])
    print('total landmark',len(icafem.landmark),'miss',len(misslandmk),misslandmk)
    
    for nodei in G.nodes():
        G.nodes[nodei]['pos'] = G.nodes[nodei]['pos'].pos
        G.nodes[nodei]['deg'] = G.degree[nodei]
        if G.nodes[nodei]['boitype']>22:
            G.nodes[nodei]['boitype'] = 0
    for edgei in G.edges():
        if G.edges[edgei]['vestype'][12]>0:
            print('merge m23')
            G.edges[edgei]['vestype'][5] += G.edges[edgei]['vestype'][12]
            G.edges[edgei]['vestype'][12] = 0
        if G.edges[edgei]['vestype'][13]>0:
            G.edges[edgei]['vestype'][6] += G.edges[edgei]['vestype'][13]
            G.edges[edgei]['vestype'][13] = 0
    nx.write_gpickle(G, picklegraphname)
    print('Graph saved',picklegraphname,'Node',len(G.nodes),'Edges',len(G.edges))
    return G

def sepset(picklelist):
    idlist = [os.path.basename(i).split('_')[3][:-4] for i in picklelist]
    setlist = list(set(idlist))
    np.random.shuffle(setlist)
    trainvalsep = int(round(len(setlist)*0.7))
    valtestsep = int(round(len(setlist)*0.85))
    trainset = setlist[:trainvalsep]
    valset = setlist[trainvalsep:valtestsep]
    testset = setlist[valtestsep:]
    trainlist = [n for i,n in enumerate(picklelist) if idlist[i] in trainset]
    vallist = [n for i,n in enumerate(picklelist) if idlist[i] in valset]
    testlist = [n for i,n in enumerate(picklelist) if idlist[i] in testset]
    return trainlist,vallist,testlist

