import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def GenRTCohsSample(dist,ps,zs,prop,delta_t,stop_val,stop_cutoff,\
    size,remove_up026=False):
    distf=dist/dist.sum()*prop.reshape(-1,1)*ps.reshape(-1,1)
    distf/=distf.sum()
    
    desk=np.zeros((*distf.shape,2))
    for i,p in enumerate(ps):
        for j in range(int((stop_val-stop_cutoff)//delta_t)):
            desk[i,j,0]=abs(zs[i])
            desk[i,j,1]=j*delta_t
    len_cohstime=np.prod(desk.shape[:2])

    desk_flatten=desk.reshape((len_cohstime,desk.shape[-1]))
    desk_idx=np.random.choice(np.arange(len_cohstime),p=distf.flatten(),size=size,replace=True)
    df=pd.DataFrame(np.take(desk_flatten,desk_idx,axis=0),columns=['cohs','rt'])
    
    if remove_up026:
        df=df[df['cohs']!=0.26]
    
    return df

def MinMaxMeanDiff(seq1,seq2):
    seq=seq1+seq2
    for i,s in enumerate(reversed(seq)):
        if s>0:
            maxs=len(seq)-i-1
            break
    mins=np.argmax(seq>0)
    mean_seq1=np.sum(seq1*(np.arange(seq1.shape[0])-mins)/(maxs-mins))/seq1.sum()
    
    mean_seq2=np.sum(seq2*(np.arange(seq2.shape[0])-mins)/(maxs-mins))/seq2.sum()
    
    return mean_seq1-mean_seq2

def ColorScaler(val):
    return (255*MinMaxScaler().fit_transform(val.reshape(-1,1))).flatten()