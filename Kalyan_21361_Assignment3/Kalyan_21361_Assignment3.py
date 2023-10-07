import numpy as np
import scipy
from matplotlib import pyplot as plt
import os

if not os.path.exists('./plots/'):
    os.makedirs('./plots/')

def filter_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    filtered = []
    for i,each in enumerate(lines):
        if i==0:
            continue
        new_data = each.split('\t')
        #If wanted to filter using gene_symbol use the below
        # new_data[-3]=='':
        #     continue
        filtered.append(new_data[1:49])
    #print(len(filtered))
    filtered=np.array(filtered).astype(float)

    return filtered


def get_N_and_D_matrices():
    N=[[[1,0,0,1]]*12,[[1,0,1,0]]*12,[[0,1,0,1]]*12,[[0,1,1,0]]*12]
    reshaped_N = np.array(N).reshape(48, 4)
    D=[[[0,1,0,0]]*12,[[1,0,0,0]]*12,[[0,0,0,1]]*12,[[0,0,1,0]]*12]
    reshaped_D = np.array(D).reshape(48, 4)
    return reshaped_N,reshaped_D

def get_p_values(data_array):
    N,D=get_N_and_D_matrices()
    list_of_p_values=[]
    rankN=np.linalg.matrix_rank(N)
    rankD=np.linalg.matrix_rank(D)
    #print(rankD,rankN)
    constant=(1/(rankD-rankN))/(1/(48-rankD))
    identity_matrix=np.identity(48)
    pseudo_inverse_term_N=np.linalg.pinv(np.matmul(np.transpose(N),N))
    pseudo_inverse_term_D=np.linalg.pinv(np.matmul(np.transpose(D),D))
    sub_term_N=identity_matrix-np.matmul(N,np.matmul(pseudo_inverse_term_N,np.transpose(N)))
    sub_term_D=identity_matrix-np.matmul(D,np.matmul(pseudo_inverse_term_D,np.transpose(D)))
    for row in data_array:
        numerator=np.matmul(np.transpose(row),np.matmul(sub_term_N,row))
        denominator=np.matmul(np.transpose(row),np.matmul(sub_term_D,row))+(1e-7)
        f= constant * ((numerator/denominator)-1)
        val=scipy.stats.f.cdf(f,(rankD-rankN),(48-rankD))
        p_val=1-val
        list_of_p_values.append(p_val)
    return list_of_p_values


def ANOVA(data_path):
    filtered_data=filter_data(data_path)
    p_values=get_p_values(filtered_data)
    #print(p_values)
    plt.hist(p_values,bins=15)
    plt.xlabel("p-values")
    plt.ylabel("Frequency")
    plt.savefig('./plots/histogram.png')

if __name__=="__main__":
    ANOVA("../data/Raw Data_GeneSpring.txt")
