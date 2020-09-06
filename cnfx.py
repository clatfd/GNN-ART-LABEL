import numpy as np
import matplotlib.pyplot as plt
class cnfx:
    def __init__(self, matrix = None, size = None):
        assert matrix is not None or size is not None
        if  matrix is None:
            self.matrix = np.zeros((size,size))
            self.size = size
        else:
            assert matrix.shape[0]==matrix.shape[1]
            self.matrix = matrix
            self.size = self.matrix.shape[0]
    def __add__(self,cnfx2):
        assert self.size == cnfx2.size
        cmatrix = self.matrix + cnfx2.matrix
        return cnfx(cmatrix)
    def __mul__(self,scale):
        cmatrix = self.matrix*scale
        return cnfx(cmatrix)
    def __truediv__(self,scale):
        cmatrix = self.matrix/scale
        return cnfx(cmatrix)
    
    def TP(self):
        return np.sum([self.matrix[i][i] for i in range(self.size)])

    def acc(self):
        return self.TP()/np.sum(self.matrix)
    
    def wpr(self):
        #wrong pred ratio
        return (np.sum(self.matrix)-self.TP())/(self.TP()-self.matrix[0][0])
    
    def solveall(self):
        if np.sum(self.matrix)==self.TP():
            return True
        else:
            return False

    def metrictype(self,ctype):
        tSUM = np.sum(self.matrix)
        tTP = self.matrix[ctype][ctype]
        tTN = tSUM-np.sum(self.matrix[ctype])-np.sum(self.matrix[:,ctype])+tTP
        tFP = np.sum(self.matrix[:,ctype])-tTP
        tFN = np.sum(self.matrix[ctype])-tTP
        typeacc = (tTP+tTN)/tSUM
        typepre = tTP/(tTP+tFP)
        typerecall = tTP/(tTP+tFN)
        return typeacc,typepre,typerecall

    def metrictypenonz(self,ctype):
        tSUM = np.sum(self.matrix[1:,1:])
        tTP = self.matrix[ctype][ctype]
        tTN = tSUM-np.sum(self.matrix[ctype,1:])-np.sum(self.matrix[1:,ctype])+tTP
        tFP = np.sum(self.matrix[1:,ctype])-tTP
        tFN = np.sum(self.matrix[ctype,1:])-tTP
        typeacc = (tTP+tTN)/tSUM
        typepre = tTP/(tTP+tFP)
        typerecall = tTP/(tTP+tFN)
        return typeacc,typepre,typerecall

    def solvecow(self):
        for ki in [3,4,5,6,18,19,20,21,22]:
            if np.sum(self.matrix[ki])!=self.matrix[ki][ki]:
                return False
        return True

    def solvekey(self):
        for ki in [1,2,3,4,5,6,7,8,17,18,19,20]:
            if np.sum(self.matrix[ki])!=self.matrix[ki][ki]:
                return False
        return True
    
    def add(self,gt,pred,ct=1):
        assert gt<self.size
        assert pred<self.size
        self.matrix[gt,pred]+=ct

    def plot(self,title=None,rmzero=False,hidetype=[]):
        plt.figure(figsize=(6,6))
        cmatrix = self.matrix.copy()
        for hi in sorted(hidetype)[::-1]:
            cmatrix = np.delete(cmatrix, hi, 0)
            cmatrix = np.delete(cmatrix, hi, 1)
        if rmzero:
            nonzmatrix = self.matrix.copy()
            nonzmatrix[0,0] = 0
            cmatrix[0,0] = np.max(nonzmatrix)
        plt.imshow(cmatrix)
        plt.colorbar()
        plt.xlabel('Prediction')
        plt.ylabel('Ground truth')
        plt.xticks(np.linspace(-0.5,self.size-len(hidetype)-0.5,self.size-len(hidetype)+1).tolist(),labels=np.arange(self.size-len(hidetype)).tolist())
        plt.yticks(np.linspace(-0.5,self.size-len(hidetype)-0.5,self.size-len(hidetype)+1).tolist(),labels=np.arange(self.size-len(hidetype)).tolist())
        if title is not None:
            plt.title(title)