DESKTOPdir = '//DESKTOP2/'
import os
import collections
import itertools
import time
import pickle
import copy
from itertools import permutations
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
from collections import Counter

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
        
def generate_graph(picklegraphname,data_aug=False,test=False):
    graphbasename = os.path.basename(picklegraphname)
    if graphbasename in graphcache.keys():
        cgraph = copy.deepcopy(graphcache[graphbasename])
    else:
        if not os.path.exists(picklegraphname):
            print(picklegraphname)
            assert os.path.exists(picklegraphname)
        else:
            #print('existing',picklegraphname)
            pjname = all_db['train'][ti].split('/')[-2]
            casename = all_db['train'][ti].split('/')[-1][:-7].split('_')[1]
            if casename not in crop_info or 'fram' not in crop_info[casename] or 'age' not in crop_info[casename]:
                print(casename)
                return
            cgraph = read_pickle_graph(picklegraphname)
            cgraph.fram = crop_info[casename]['fram']
            cgraph.age = crop_info[casename]['age']
            cgraph.pjname = casename
            
            cgraph.db = picklegraphname.split('/')[-2]
            cgraph.name = picklegraphname

            #norm pos
            mean_val_off_L = np.array([0.55649313, 0.40892808, 0.1999954])
            mean_val_off_R = np.array([0.40247367, 0.41225129, 0.20614317])
            #UNC
            #mean_val_off_L = np.array([249.7012,     189.37026667,  38.30013333])
            #mean_val_off_R = np.array([196.35006667, 189.16006667,  38.1498])
            mean_val_off_LR = (mean_val_off_L+mean_val_off_R)/2
            #mpos = [220.125,   184.764 ,   23.33215]
            
            if 'CROPCheck' in picklegraphname:
                res = 0.3515625
            elif 'ArizonaCheck' in picklegraphname:
                res = 0.3906
            elif 'BRAVE' in picklegraphname:
                res = 0.4297
            elif 'Parkinson2TPCheck' in picklegraphname:
                res = 0.399120
            elif 'UNC' in picklegraphname:
                res = 0.51339286
            elif 'Anzhen' in picklegraphname:
                res = 0.469
            else:
                res = 1
                print('no db',picklegraphname)
            
            if test == True:
                mpos = mean_val_off_LR
            else:
                lapos_L = [cgraph.nodes[i]['pos'] for i in cgraph.nodes() if cgraph.nodes[i]['boitype'] in [3]]
                if len(lapos_L)==0:
                    mpos_L = mean_val_off_L
                    print("ICAL missing",picklegraphname)
                else:
                    mpos_L = np.mean(lapos_L,axis=0)*res/200
                lapos_R = [cgraph.nodes[i]['pos'] for i in cgraph.nodes() if cgraph.nodes[i]['boitype'] in [4]]
                if len(lapos_R)==0:
                    mpos_R = mean_val_off_R
                    print("ICAR missing",picklegraphname)
                else:
                    mpos_R = np.mean(lapos_R,axis=0)*res/200
                mpos = (mpos_L+mpos_R)/2
                #print('mpos',mpos)
                
            for i in cgraph.nodes():
                cpos = copy.copy(cgraph.nodes[i]['pos'])
                npos = (np.array(cpos))*res/200-np.array(mpos)
                if 'UNC' in picklegraphname:
                    if cgraph.nodes[i]['pos'][2]<10:
                        npos -= [0,0,50*res/200]
                        #print('low 10',npos)
                    offset = [-0.09209063, -0.07075882,  0.11279931]
                    npos += offset
                cgraph.nodes[i]['pos'] = npos.tolist()
                cgraph.nodes[i]['rad'] = cgraph.nodes[i]['rad']*res
            for i in cgraph.edges():
                cgraph.edges[i]['rad'] *= res
                cgraph.edges[i]['dist'] *= res
                    
        graphcache[graphbasename] = copy.deepcopy(cgraph)
    if data_aug:
        cgraph = auggraph(cgraph,0.1)
    return cgraph

def auggraph(cgraph,offset):
    offsamples = [offset/10*i for i in range(-10,11,1)]
    offpos = randaug.choice(offsamples,3)
    #print(offpos)
    for i in cgraph.nodes():
        cpos = copy.copy(cgraph.nodes[i]['pos'])
        npos = np.array(cpos) + offpos
        cgraph.nodes[i]['pos'] = npos.tolist()
    return cgraph

def read_pickle_graph(picklegraphname):
    return nx.read_gpickle(picklegraphname)
    
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
    icafem.xml.getlandmark(IGNOREM3=1)
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
    for i,j in icafem.xml.landmark:
        #print(icafem.VesselName[i],j)
        if j.hashpos() not in icafem.simghash:
            #print(icafem.VesselName[i],'Not Found')
            misslandmk.append(icafem.NodeName[i])
    print('total landmark',len(icafem.xml.landmark),'miss',len(misslandmk),misslandmk)
    
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

def prepare_graphs(dbnames):
    simplerandomsep = 1
    alllist = {'train':[],'val':[],'test':[]}
    for dbname in dbnames:
        icafedir = DESKTOPdir+'/iCafe/result/'+dbname
        #icafedir = DESKTOPdir + '/iCafe/result/'+dbname
        if not os.path.exists(icafedir):
            print('no exist',icafedir)
            continue 
        dblistfile = icafedir+'/db.list'
        if os.path.exists(dblistfile):
            print('Load from db list',dblistfile)
            with open (dblistfile, 'rb') as fp:
                picklegraphdict = pickle.load(fp)
            for key in picklegraphdict.keys():
                for di,fi in enumerate(picklegraphdict[key]):
                    dbname = fi.split('/')[-3]
                    casename = fi.split('/')[-2]
                    picklegraphname = generate_graph_icafe(dbname,casename)
                    assert picklegraphname is not None
                    picklegraphdict[key][di] = picklegraphname
            
        else:
            icafelist = os.listdir(icafedir)[:]
            #remove non dir
            for di in icafelist[::-1]:
                if not os.path.isdir(icafedir+'/'+di) or len(di.split('_'))!=3:
                    del icafelist[icafelist.index(di)]
            picklegraphlist = []
            for icafefoldername in icafelist:
                picklegraphname = generate_graph_icafe(os.path.basename(icafedir), icafefoldername, graphtype = 'graphsim')
                if picklegraphname is not None:
                    picklegraphlist.append(picklegraphname)
            if simplerandomsep:
                np.random.shuffle(picklegraphlist)
                picklegraphdict = {}
                trainvalsep = int(round(0.7*len(picklegraphlist)))
                valtestsep = int(round(0.85*len(picklegraphlist)))
                picklegraphdict['train'] = picklegraphlist[:trainvalsep]
                picklegraphdict['val'] = picklegraphlist[trainvalsep:valtestsep]
                picklegraphdict['test'] = picklegraphlist[valtestsep:]
            else:
                picklegraphdict = {}
                picklegraphdict['train'],picklegraphdict['val'],picklegraphdict['test'] = sepset(picklegraphlist)
            #save list separation
            with open (dblistfile, 'wb') as fp:
                pickle.dump(picklegraphdict,fp)
            print('Save to db list',dblistfile)
        for key in picklegraphdict.keys():
            print(key,'len',len(alllist[key]),len(picklegraphdict[key]))
            alllist[key].extend(picklegraphdict[key])
            
    return alllist