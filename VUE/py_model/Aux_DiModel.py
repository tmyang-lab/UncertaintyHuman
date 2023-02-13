import numpy as np
aa=np.asarray
from numba import njit,jit,prange

@njit
def binary_search(func,a,b,target=0.05,bar=1e-10):
    
    low=a;high=b
    lowf=func(a);highf=func(b)
    if abs(lowf-target)<bar:return a
    if abs(highf-target)<bar:return b
    if not ((lowf<target and target<highf) or (lowf>target and target>highf)):
        print("k might be too small, cannot spilit it up")
        return b


    while low<=high:
        mid=(low+high)/2
        midf=func(mid)
        if abs(midf-target)<bar:
            return mid
        if (lowf<target and midf>target) or (lowf>target and midf<target):
            high=mid;highf=midf
        if (highf<target and midf>target) or (midf>target and highf<target):
            low=mid;lowf=midf
    raise Exception("binary search cannot find the target value")

@njit
def big_Z_bound(k,zs,ps,var,x,t,mid_z=9):
    """
    for probility on z range.
    """
    prob_z=np.zeros_like(zs)
    C=k/(2*var)
    x2=2*x
    kt=k*t
    
    for r in prange(zs.shape[0]):
        tmp=0.
        for i in prange(zs.shape[0]):
            if i!=r:
                tmp=tmp+ps[i]/ps[r]*np.exp(C*(zs[i]-zs[r])*(x2-kt*(zs[i]+zs[r])))
        prob_z[r]=1/(1+tmp)
    if zs.shape[0]%2==0: # even
        prob_left=np.sum(prob_z[:mid_z])
    else: # odd
        prob_left=np.sum(prob_z[:mid_z])+prob_z[mid_z]/2
    return prob_left

@njit
def big_Z_bound_noaccu(k,zs,ps,var,x,delta_t,mid_z=9):
    """
    for probility on z range.
    """
    prob_z=np.zeros_like(zs)
    C=2*delta_t*var
    x2=2*x
    kdt=k*delta_t
    
    for r in prange(zs.shape[0]):
        tmp=0
        for i in prange(zs.shape[0]):
            if i!=r:
                tmp=tmp+ps[i]/ps[r]*np.exp(kdt*(zs[i]-zs[r])*(x2-kdt*(zs[i]+zs[r]))/C)
        prob_z[r]=1/(1+tmp)
    if zs.shape[0]%2==0: # even
        prob_left=np.sum(prob_z[:mid_z])
    else: # odd
        prob_left=np.sum(prob_z[:mid_z])+prob_z[mid_z]/2
    return prob_left

# @njit
def transfer_D_to_bounds(D):

    upp=-np.ones(D.shape[1],dtype=np.int64)
    low=-np.ones(D.shape[1],dtype=np.int64)
    inn_upp=-np.ones(D.shape[1],dtype=np.int64)
    inn_low=-np.ones(D.shape[1],dtype=np.int64)

    waitp=D[1:]==2.
    waitm=D[:-1]==2.
    rightp=D[1:]==1.
    leftm=D[:-1]==0.

    val,idx=np.where(waitp & leftm)
    low[idx]=val

    val,idx=np.where(waitm & rightp)
    upp[idx]=val+1

    val,idx=np.where((D[1:]==3.) & (waitm|leftm))
    inn_low[idx]=val+1
    
    val,idx=np.where((D[:-1]==3.) & (waitp|rightp))
    inn_upp[idx]=val

    low[-1]=low[-2]
    upp[-1]=upp[-2]
    inn_low[-1]=inn_low[-2]
    inn_upp[-1]=inn_upp[-2]
    
    return upp,low,inn_upp,inn_low

def dist_mean_std(t_vec,dist,z_side):
    g_mu=np.zeros_like(z_side);g_sigma=np.zeros_like(z_side)
    for i,z in enumerate(z_side):
        g_mu[i]=np.sum(t_vec*dist[i])/dist[i].sum()
        g_sigma[i]=np.sum(dist[i]/dist[i].sum()*(t_vec-g_mu[i])**2)
    return aa(g_mu),aa(g_sigma)

def cr_coh(t_vec,g_uppers,g_lowers,g_uppers2,g_lowers2,g_uppers_std,g_lowers_std,zs,mid_z,phymetric=False):
    crc_std=np.zeros_like(zs)
    crc_opt_lr=np.zeros_like(zs)
    crc_opt_lr2=np.zeros_like(zs)
    for i,z in enumerate(zs):
        if i>mid_z or phymetric:
            crc_std[i]=np.nansum(g_uppers_std[i])/((np.nansum(g_uppers_std[i])+np.nansum(g_lowers_std[i])))
            crc_opt_lr[i]=np.nansum(g_uppers[i])/((np.nansum(g_uppers[i])+np.nansum(g_lowers[i])))
            crc_opt_lr2[i]=np.nansum(g_uppers2[-1])/((np.nansum(g_uppers2[-1])+np.nansum(g_lowers2[-1])))
        elif i<mid_z:
            crc_std[i]=np.nansum(g_lowers_std[i])/((np.nansum(g_uppers_std[i])+np.nansum(g_lowers_std[i])))
            crc_opt_lr[i]=np.nansum(g_lowers[i])/((np.nansum(g_uppers[i])+np.nansum(g_lowers[i])))
            crc_opt_lr2[i]=np.nansum(g_lowers2[0])/((np.nansum(g_uppers2[0])+np.nansum(g_lowers2[0])))
        else:
            crc_std[i]=np.nansum(g_uppers_std[i]/(np.nansum(g_uppers_std[i])+np.nansum(g_lowers_std[i])))
            crc_opt_lr[i]=np.nansum(g_uppers[i]/(np.nansum(g_uppers[i])+np.nansum(g_lowers[i])))
            crc_opt_lr2[i]=np.nansum(g_lowers2[0]/(np.nansum(g_uppers2[0])+np.nansum(g_lowers2[0])))
    return crc_std,crc_opt_lr,crc_opt_lr2

if __name__=='__main__':
    D=np.load('test_data/D.npy')
    bd=transfer_D_to_bounds(D)
    np.save('test_data/boundsD',bd)
