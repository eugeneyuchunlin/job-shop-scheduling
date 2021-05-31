from cProfile import label
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt # import matplotlib相關套件
import matplotlib.patches as mpatches

class machine(): #answer map
    def __init__(self): 
        self.start = []
        self.end = []
        self.assignjob = []
        self.urgent = []
    def __repr__(self):
        return repr((self.start, self.end, self.assignjob, self.urgent))

if __name__ == '__main__':
    data = pd.read_excel('data.xlsx',sheet_name= [0,1,2,3], engine="openpyxl")
    f = open('result.txt', 'r')
    Mach = []
    count = 0
    for i in range(10):
        Mach.append(machine())
    for line in f.readlines():
        tmp = line.split()
        Mach[int(tmp[1])-1].assignjob.append(int(tmp[0]))
        Mach[int(tmp[1])-1].start.append(float(tmp[2]))
        Mach[int(tmp[1])-1].end.append(float(tmp[3]))
    color_barh = ['r','g','b','c','m','y','navy','coral', 'brown', 'orange','black']
    recipe_label = ['X1000','Y2000','Z3000','X4000','Y5000','Z6000','X7000','Y8000','Z9000','X0000','scrapped']
    first_label = [0]*11
    for i in range(10):
        for j in range(len(Mach[i].assignjob)): 
            if (data[0].values[Mach[i].assignjob[j]-1][4]*60) < Mach[i].start[j]:
                colorsave = 'black'
            else:
                colorsave = color_barh[int(data[0].values[Mach[i].assignjob[j]-1][8][1])]
            plt.barh(i,Mach[i].end[j]-Mach[i].start[j],left=Mach[i].start[j], color = colorsave)
            plt.text(Mach[i].start[j]+(Mach[i].end[j]-Mach[i].start[j])/4,i,'%s'%(Mach[i].assignjob[j]),color="white")
    f.close()
    patches = [mpatches.Patch(color=color_barh[i], label="{:s}".format(recipe_label[i]) ) for i in range(11) ]
    plt.yticks(np.arange(10),np.arange(1,11))
    plt.xlabel("Time(min)")
    plt.ylabel("Machine ID")
    plt.legend(handles=patches)
    plt.show()
