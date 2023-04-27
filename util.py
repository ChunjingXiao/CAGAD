import dgl

def points2txt(data,roi_file):
    f = open(roi_file, 'a')
    f.write(str(float('%.4f' % data)))
    f.write("\n")
    f.close()

def saveOneStep(graph,index:list,label):
    # obtain neighbors of anomalies
    H = []
    for i in index:
        # i = i + len(labels)//2
        t = list(dgl.bfs_nodes_generator(graph,i,True))
        count = 0
        temp = [1]
        if (len(t)>2):
            temp =  t[1].numpy().tolist()
            for i in temp:
                if label[i]==0:
                   count +=1
        else:
            count = 0
        s = count/len(temp)
        points2txt(s,"normalRatio.txt")
    
