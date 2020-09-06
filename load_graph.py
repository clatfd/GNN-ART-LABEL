import pickle
import os
import copy
import networkx as nx
import numpy as np

def prepare_graphs(dbnames,graphfoldername):
    simplerandomsep = 1
    alllist = {'train':[],'val':[],'test':[]}
    for dbname in dbnames:
        dblistfile = graphfoldername+'/'+dbname+'/db.list'
        if os.path.exists(dblistfile):
            print('Load from db list',dblistfile)
            with open (dblistfile, 'rb') as fp:
                picklegraphdict = pickle.load(fp)
            for key in picklegraphdict.keys():
                for di,fi in enumerate(picklegraphdict[key]):
                    picklegraphname = graphfoldername+'/'+fi
                    assert picklegraphname is not None
        else:
            print('generate data set separation file and put under graph folder',dblistfile)
        for key in picklegraphdict.keys():
            print(key,'len',len(alllist[key]),len(picklegraphdict[key]))
            for case in picklegraphdict[key]:
                alllist[key].append(graphfoldername+'/'+case)        
    return alllist


def read_pickle_graph(picklegraphname):
    return nx.read_gpickle(picklegraphname)

def auggraph(cgraph,offset,randaug):
    offsamples = [offset/10*i for i in range(-10,11,1)]
    offpos = randaug.choice(offsamples,3)
    #print(offpos)
    for i in cgraph.nodes():
        cpos = copy.copy(cgraph.nodes[i]['pos'])
        npos = np.array(cpos) + offpos
        cgraph.nodes[i]['pos'] = npos.tolist()
    return cgraph

def generate_graph(graphcache,picklegraphname,randaug,data_aug=False,test=False):
    graphbasename = os.path.basename(picklegraphname)
    if graphbasename in graphcache.keys():
        cgraph = copy.deepcopy(graphcache[graphbasename])
    else:
        if not os.path.exists(picklegraphname):
            print(picklegraphname)
            assert os.path.exists(picklegraphname)
        else:
            #print('existing',picklegraphname)
            cgraph = read_pickle_graph(picklegraphname)
            cgraph.db = picklegraphname.split('/')[-2]
            cgraph.name = picklegraphname

            #norm pos
            mean_val_off_L = np.array([0.55649313, 0.40892808, 0.1999954])
            mean_val_off_R = np.array([0.40247367, 0.41225129, 0.20614317])
            #UNC
            #mean_val_off_L = np.array([249.7012,     189.37026667,  38.30013333])
            #mean_val_off_R = np.array([196.35006667, 189.16006667,  38.1498])
            mean_val_off_LR = (mean_val_off_L+mean_val_off_R)/2
            
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
                        print('low 10',npos)
                    offset = [-0.09209063, -0.07075882,  0.11279931]
                    npos += offset
                cgraph.nodes[i]['pos'] = npos.tolist()
                cgraph.nodes[i]['rad'] = cgraph.nodes[i]['rad']*res
            for i in cgraph.edges():
                cgraph.edges[i]['rad'] *= res
                cgraph.edges[i]['dist'] *= res
                    
        graphcache[graphbasename] = copy.deepcopy(cgraph)
    if data_aug:
        cgraph = auggraph(cgraph,0.1,randaug)
    return cgraph