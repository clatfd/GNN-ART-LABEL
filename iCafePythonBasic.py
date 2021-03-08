import numpy as np
import copy
from gnn_utils import VESTYPENUM

class SnakeList():
    def __init__(self,snakelist=None):
        if snakelist is None:
            self._snakelist = []
        else:
            self._snakelist = snakelist
        self.comp_map = None
        
    def __repr__(self):
        return 'Snakelist with %d snakes' % (len(self._snakelist))

    def __len__(self):
        return len(self._snakelist)

    def __getitem__(self, key):
        return self._snakelist[key]
    
    #from snakelist to swc list
    def toSWCList(self):
        swclist = []
        for snakei in range(self.NSnakes):
            for pti in range(self._snakelist[snakei].NP):
                swclist.append(self._snakelist[snakei][pti])
        return swclist
    
    def addSnake(self,new_snake):
        self._snakelist.append(new_snake)

    @property
    def NSnakes(self):
        return len(self._snakelist)
    
    #from ves snakelist to ves list
    def toVesList(self):
        veslist = [[] for i in range(VESTYPENUM)]  # list, first of vessel type, then each snake in that type
        for snakei in range(self.NSnakes):
            ctype = self._snakelist[snakei].type
            if ctype==0 or ctype>=VESTYPENUM:
                print('unseen ctype',snakei)
            veslist[ctype].append(self._snakelist[snakei])
        return veslist


class Snake:
    def __init__(self,swcnodelist=None,type = 0):
        if swcnodelist is None:
            self.snake = []
        else:
            self.snake = copy.copy(swcnodelist)
        self.type = type
        self.id = None

    def  __repr__(self): 
        return 'Snake with %d points, type %d'%(len(self.snake),self.type)
    
    @property
    def NP(self):
        self._NP = len(self.snake)
        return self._NP
    
    def __len__(self):
        return self.NP

    def __getitem__(self, key):
        return self.snake[key]


class SWCNode(object):
    def __init__(self,cpos,crad,cid=None,ctype=None,cpid=None):
        self.id = cid
        self.type = ctype
        self.pos = cpos
        self.rad = crad
        self.pid = cpid
        #match id to another snake
        self.link_id = None

    @classmethod
    def fromline(cls, line):
        ct = line.split(' ')
        cid = int(ct[0])
        ctype = int(ct[1])
        cpos = Point3D([float(i) for i in ct[2:5]])
        crad = float(ct[5])
        cpid = int(ct[6])
        return cls(cpos,crad,cid,ctype,cpid)

    def getlst(self):
        return [self.id,self.type,self.pos.x,self.pos.y,self.pos.z,self.rad,self.pid]


class Point3D:
    def __init__(self, listinput, pointy=None, pointz=None):
        if type(listinput) in [list,np.ndarray,tuple]:
            self.x = listinput[0]
            self.y = listinput[1]
            self.z = listinput[2]
            self.pos = listinput
        elif type(listinput) in [float,int,np.float64,np.float32] and pointy is not None and pointz is not None:
            self.x = listinput
            self.y = pointy
            self.z = pointz
            self.pos = [listinput,pointy,pointz]
        else:
            print('__init__ unknown type',type(listinput))
    def  __repr__(self): 
        return 'Point3D: %.3f %.3f %.3f'%(self.x,self.y,self.z)
    def __add__(self,pt2):
        return Point3D([self.x+pt2.x,self.y+pt2.y,self.z+pt2.z])
    def __sub__(self,pt2):
        return Point3D([self.x-pt2.x,self.y-pt2.y,self.z-pt2.z])
    def __mul__(self,scale):
        if type(scale) in [float,int,np.float64,np.float32]:
            return Point3D([self.x*scale,self.y*scale,self.z*scale])
        elif type(scale) in [Point3D,'iCafe.Point3D']:
            return self.x*scale.x+self.y*scale.y+self.z*scale.z
        else:
            print('__mul__ Unsupport type',type(scale))
    def __truediv__(self,scale):
        if scale ==0:
            print(self.x,self.y,self.z,'divide Point3D by scale 0')
            scale = 1
        return Point3D([self.x/scale,self.y/scale,self.z/scale])
    def __neg__(self):
        return Point3D([-self.x,-self.y,-self.z])
    def hashPos(self):
        return '-'.join(['%.3f'%i for i in self.lst()])
    
    def lst(self):
        return [self.x,self.y,self.z]
    
    def dist(self,pt2):
        return np.linalg.norm((self-pt2).pos)
    
    def vecLenth(self):
        return np.linalg.norm(self.pos)
    
    def norm(self):
        abnorm = np.linalg.norm(self.pos)
        return Point3D(self.pos/abnorm)
    def lst(self):
        return [self.x,self.y,self.z]
    def intlst(self):
        return [int(round(self.x)),int(round(self.y)),int(round(self.z))]
    def toIntPos(self):
        self.x = int(round(self.x))
        self.y = int(round(self.y))
        self.z = int(round(self.z))
    def prod(self,pt2):
        return self.x*pt2.x+self.y*pt2.y+self.z*pt2.z
    def hashPos(self):
        return '-'.join(['%.3f'%i for i in self.lst()])
    def intHashPos(self):
        return '-'.join(['%d'%i for i in self.lst()])
    def posMatch(self,points):
        bestmatchscore = 0
        bestid = -1
        for idx,pi in enumerate(points):
            cmatchscore = self*pi
            if cmatchscore>bestmatchscore:
                bestmatchscore = cmatchscore 
                bestid = idx
        return bestid