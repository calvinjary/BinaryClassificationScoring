# E:\Dropbox\9 PhD Research\2 PyPE scoring

'''
Welcome to Binary Classification sorting, scoring and generating a precision-recall curve by Calvin Jary
'''

# Importing the libraries, note numpy is faster than pandas for this application
import pandas as pd
import numpy as np
import struct
import statistics
import os
from time import localtime, strftime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from time import localtime, strftime

###################################################### main algorithm  ######################################################################

#Results2 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 
#Results = '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''

dfGroundTruth = pd.read_csv('G.txt', header = None, skiprows = 1, sep = ', ', index_col = False, engine = 'python')
dfRandomPairs = pd.read_csv('R.txt', header = None, skiprows = 1, sep = ', ', index_col = False, engine = 'python')

counter, counter2, counter3 = int(0), int(0), int(0)

#print(type(dfRandomPairs.iloc[4,0]))

dfTotal = pd.concat([dfGroundTruth, dfRandomPairs], axis=1)

npTotal = dfTotal.values

#dfTotalSorted.sort_values(by = 3, axis = 1, ascending = True, inplace = True)
#dfTotalSorted.sort_index(axis=1, ascending = True, inplace = True, by = 3)

npResults = np.zeros((len(npTotal), 2))



# begin of testing loop
#for master in range(4, 5):
    
master = int(4)    
    
#master = int(147)

npSliver, npSliverSorted = np.zeros((2, len(npTotal[0]))), np.zeros((2, len(npTotal[0])))

for i in range(len(npTotal[0])):
    
    if npTotal[2, i] == 'G':  # ground truth is 1 
        npSliver[0, i] = 1
    elif npTotal[2, i] == 'R':  # random is 0
        npSliver[0, i] = 0
    else:
        print("encoding error!")
        
    try:
        npSliver[1, i] = npTotal[master, i]
    except:
        continue
    
npTemp = npSliver 


###################################### Sorting from lowest to highest ##################################################
MinValue, MinIndex = float(0), int(0)
for j in range(len(npTotal[0])):
    counter += 1
    MinValue = 99999
    for i in range(len(npTotal[0])):
        if npTemp[1, i] < MinValue:
            MinValue = npTemp[1, i]
            MinIndex = i
    npSliverSorted[0,j] = npTemp[0,MinIndex]
    npSliverSorted[1,j] = npTemp[1,MinIndex]
    npTemp[1, MinIndex] = 9999
    npTemp[0, MinIndex] = 9


    if (counter % 100) == 0:
        print(counter, ' counter1 has been done successfully')
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))


###################################### Scoring recall for a given precision ##################################################
TotalGroundTruth, TotalRandomPairs = int(0), int(0)
Precision, Recall = float(0), float(0)
BestRecall = float(0)



for j in range(len(npSliverSorted[0])):
    counter2 += 1
    TotalGroundTruth, TotalRandomPairs = 0, 0
    for i in range(j, len(npSliverSorted[0])):
        if (npSliverSorted[0,i] == 1):
            TotalGroundTruth += 1
        elif (npSliverSorted[0,i] == 0):
            TotalRandomPairs += 1
    try:
        Precision = TotalGroundTruth / (TotalGroundTruth + TotalRandomPairs)
    except:
        Precision = 0.1
    # choose the desired precision here!
    if Precision >= 0.90: 
        Recall = TotalGroundTruth / len(dfGroundTruth.iloc[0])
        #print('the recall is:', Recall)
        npResults[master,0] = master
        npResults[master,1] = Recall
        break

    if (counter2 % 100) == 0:
        print(counter2, ' counter2 has been done successfully')
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    


# end of testing loop



###################################### Sorting Results from lowest to highest ##################################################
#npResultsBackup = npResults
npResultsSorted = np.zeros((len(npResults), 2))

print('testing loop done, begin of sorting results!')


MinValue, MinIndex = float(0), int(0)
for j in range(len(npResults)):
    counter3 += 1
    MinValue = 99999
    for i in range(len(npResults)):
        if npResults[i, 1] < MinValue:
            MinValue = npResults[i, 1]
            MinIndex = i
    npResultsSorted[j,0] = npResults[MinIndex,0]
    npResultsSorted[j,1] = npResults[MinIndex,1]
    npResults[MinIndex,1] = 9999
    npResults[MinIndex,0] = 9

    if (counter3 % 100) == 0:
        print(counter3, ' counter3 has been done successfully')
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    

# normalizing for actual result numbers
for i in range(len(npResultsSorted)):
    npResultsSorted[i,0] = (npResultsSorted[i,0] - 3) 




###################################### printing results ##################################################
Results_PandasDataframe = pd.DataFrame(npResultsSorted)
Results_PandasDataframe.to_csv('scoringresults.txt', sep = '\t', index = False, index_label = False)  







###################################### Generating a Precision Recall Curve ##################################################

score = npSliverSorted[1,:]  # the measured score
y = npSliverSorted[0,:]      # whether the patient has ebola or not

roc_x = []
roc_y = []
min_score = min(score)
max_score = max(score)
thr = np.linspace(min_score, max_score, 10000)
FP=float(0.0000000000000000001)      # (to avoid divide by zero errors)
FN=float(0.0000000000000000001)      # (to avoid divide by zero errors)
TP=float(0)
N = sum(y)
P = len(y) - N


for (i, T) in enumerate(thr):
    for i in range(0, len(score)):
        if (score[i] > T):
            if (y[i]==1):
                TP += 1
            elif (y[i]==0):
                FP += 1
        elif (score[i] < T):
            if (y[i]==1):
                FN += 1                

    roc_x.append(TP/(TP+float(FN)))                
    roc_y.append(TP/(TP+float(FP)))  
 
    FP=0.0000000000000000001        # (to avoid divide by zero errors)
    FN=0.0000000000000000001        # (to avoid divide by zero errors)
    TP=0



plt.scatter(roc_x2, roc_y2, c = 'green', linewidths = 0, marker = '.')
#plt.scatter(0.9, 0.6036, linewidths = 10, marker = 'X')   # part iv)


plt.scatter(roc_x, roc_y, c = 'red', linewidths = 0, marker = '.')
#plt.scatter(0.9, 0.6036, linewidths = 10, marker = 'X')   # part iv)
plt.title('precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()



#roc_x2 = roc_x
#roc_y2 = roc_y




