
import numpy as np
from numba import njit,jit,prange
import matplotlib.pyplot as plt
import math

aa=np.asarray

@njit
def jitmeshgrid(x,y):
    x_grid=np.zeros((x.shape[0],y.shape[0]))
    y_grid=np.zeros((x.shape[0],y.shape[0]))
    for i in prange(y.shape[0]):
        x_grid[:,i]=x
    for j in prange(x.shape[0]):
        y_grid[j]=y
    return x_grid,y_grid

@njit
def jitmaxargmax(x,axis=0):
    """max,argmax for 2d array using numba

    Parameters
    ----------
    x : ndarray(2d)
        supposed to a 2d array
    axis : int
        
    """
    if axis==0:
        ret=np.zeros(x.shape[1])
        arg_ret=np.zeros(x.shape[1],dtype=np.int64)
        for i in prange(x.shape[1]):
            arg_ret[i]=np.argmax(x[:,i])
            ret[i]=x[arg_ret[i],i]
            
    elif axis==1:
        ret=np.zeros(x.shape[0])
        arg_ret=np.zeros(x.shape[0],dtype=np.int64)
        for i in prange(x.shape[0]):
            arg_ret[i]=np.argmax(x[i])
            ret[i]=x[arg_ret[i],i]

    return ret,arg_ret

@njit
def jitsum(x,axis=0):
    """ sum for 3d array using numba

    Parameters
    ----------
    x : ndarray(3d)
        supposed to a 2d array
    axis : int
        
    """    
    if axis==0:
        I,J=x.shape[1],x.shape[2]
        ret=np.zeros((I,J))
        for i in prange(I):
            for j in prange(J):
                ret[i,j]=np.sum(x[:,i,j])
            
    elif axis==1:
        I,J=x.shape[0],x.shape[2]
        ret=np.zeros((I,J))
        for i in prange(I):
            for j in prange(J):
                ret[i,j]=np.sum(x[i,:,j])
    elif axis==2:
        I,J=x.shape[0],x.shape[1]
        ret=np.zeros((I,J))
        for i in prange(I):
            for j in prange(J):
                ret[i,j]=np.sum(x[i,j,:])
    return ret

@njit
def jitaddouter(x,dx):
    ret=np.zeros((x.shape[0],dx.shape[0]))
    for i in range(x.shape[0]):
        for j in range(dx.shape[0]):
            ret[i,j]=x[i]+dx[j]
    return ret


@njit
def findnearest(values,vec):
    """find an array of values is nearest in a specified array.

    Parameters
    ----------
    value : np.ndarray
        source value
    vec : np.ndarray
        target position array, i.e. the grided array

    Returns
    -------
    np.ndarray
        positions
    """
    indices=np.zeros_like(values,dtype=np.int64)
    for i in prange(values.shape[0]):
        for j in prange(values.shape[1]):
            indices[i,j]=int(np.argmin(np.abs(vec-values[i,j])))
    return indices

# @njit
# def abolish_xdx(xdx,num_states):
#     for i in range(xdx.shape[0]):
#         xdx[i][(xdx[i]==0) & (np.roll(xdx[i],-1)==0)]=-1
#         xdx[i][(xdx[i]==num_states-1) & (np.roll(xdx[i],-1)==num_states-1)]=-1
#     return xdx

from scipy.stats import norm as normal


@njit(fastmath=True)
def big_Z(k,zs,ps,var,x_grids,t_grids):
    prob_z=np.zeros((zs.shape[0],x_grids.shape[0],x_grids.shape[1]))
    C=k/(2*var)
    x2_grids=2*x_grids
    kt_grids=k*t_grids
    
    for r in prange(zs.shape[0]):
        tmp=np.zeros_like(prob_z[r])
        for i in prange(zs.shape[0]):
            if i!=r:
                tmp=tmp+ps[i]/ps[r]*np.exp(C*(zs[i]-zs[r])*(x2_grids-kt_grids*(zs[i]+zs[r])))
        prob_z[r]=1/(1+tmp)
    # assert(not (prob_z==0).any())
    return prob_z

# @jit(fastmath=True, parallel=True)
def dx_under_z(k,dx_vec,zs,delta_t,var):
    
    prob_dx_under_z=np.zeros((zs.shape[0],dx_vec.shape[0]))
    
    for i in prange(zs.shape[0]):
        prob_dx_under_z[i]=normal.pdf((dx_vec-k*zs[i]*delta_t)/np.sqrt(delta_t*var))
        prob_dx_under_z[i]=prob_dx_under_z[i]/np.sum(prob_dx_under_z[i])
    
    return prob_dx_under_z

@njit
def dx_under_x(prob_dx_under_z,prob_z):
    prob_dx_under_x=np.zeros(
        (prob_dx_under_z.shape[1],prob_z.shape[1],prob_z.shape[2]))
    for i in prange(prob_z.shape[0]):
        for j in prange(prob_dx_under_z.shape[1]):
            prob_dx_under_x[j]+=prob_dx_under_z[i,j]*prob_z[i]
    return prob_dx_under_x # (dx,x,t)

@njit
def expectation(V,prob_dx_under_x,xp1_state,t):
    E=np.zeros(xp1_state.shape[0])
    
    for i,x in enumerate(xp1_state):
        av_x=x!=-1
        E[i]=np.sum(V[x[av_x],t+1]*prob_dx_under_x[av_x,i,t])
    return E

@njit
def max_(x):
    
    Vt=np.zeros(x.shape[1])
    Dt=np.zeros(x.shape[1],dtype=np.int64)
    Vt,Dt=jitmaxargmax(x)
    # 0,1 is for the 1,2 accumulator; 2 if exists, for uncertainty option; 3 if exists, for waiting
    Dt[(Dt==0) & (x[0]==x[1])]=12 # two accumulator have same value'

    return Vt,Dt


def d_prob(k,a,c,rho,tw,tu,EV2,zs,ps,sigma,num_states,delta_t,xbar=np.nan,stop_val=2,ana=np.nan,mapping_range=3):
    """discrete prior optimal policy model 

    Parameters
    ----------
    k : float
        drift rate
    a : float or ndarray
        reward on left or right choice
    c : float
        cost per time unit
    rho : float
        average reward
    tw : float
        time for waiting
    tu : float 
        time for switching to second stage
    EV2 : float
        expected reward of stage 2
    zs : ndarray[float]
        coherence strength, zs is supposed to be symmetric
    ps : ndarray[float]
        probability
    sigma : float
        standard deviation of evidence
    num_states : int
        number of states on x(evidence)
    delta_t : float
        unit time interval
    xbar : float
        maximum x value, by default np.nan
    stop_val : int, optional
        end of the time, by default 2
    ana : float or array
        another reward value, by default np.nan
    mapping_range : int, optional
        times of sigma on x, by default 3

    Returns
    -------
        V0 : float
            expected value at the beginning of stage 1
        V : ndarray[float]
            value function under (t,x)-space
        D : ndarray[float]
            decision made in (t,x)-space
            0- left
            1- right
            2- waiting for the next evidence
            3- uncertain response
        EVnext : ndarray[float]
            expected value function under (t,x)-space
        rho : float
            average reward
    """


    tbar=np.floor(stop_val/delta_t)
    t_vec=np.arange(delta_t,tbar*delta_t+1e-10,delta_t)
    
    # assert(a.shape[0]==t_vec.shape[0])
    var=sigma**2
    

    T=len(t_vec)

    z_max=np.max(zs)
    z_min=np.min(zs)
    mid_state=int(np.floor(num_states/2))
    mid_z=int(np.floor(zs.shape[0]/2))

    sigma_range=mapping_range*sigma
    

    upper=k*z_max*stop_val+sigma_range if np.isnan(xbar) else xbar 
    lower=-upper

    dx=(upper-lower)/(num_states-1)

    dx_vec=np.concatenate((
        np.flip(np.arange(0,-sigma_range-1e-14,-dx))[:-1],
        np.arange(0,sigma_range+1e-14,dx)))

    x_vec=np.r_[np.flip(np.arange(0,lower-1e-14,-dx))[:-1],np.arange(0,upper+1e-14,dx)]
    x_grids,t_grids=jitmeshgrid(x_vec,t_vec)
    xp1_mat=jitaddouter(x_vec,dx_vec)
    xp1_state=findnearest(xp1_mat,x_vec) #(x,dx)

    EV2c=(EV2-rho*tu)
    EV2c_vec=EV2c*np.ones(num_states)

    # xp1_state=abolish_xdx(xp1_state,num_states)

    # prob preparation
    prob_z=big_Z(k,zs,ps,var,x_grids,t_grids)
    # prob left and right
    if zs.shape[0]%2==0: # even
        prob_left,prob_right=jitsum(prob_z[:mid_z]),jitsum(prob_z[mid_z:])
    else: # odd
        prob_left,prob_right=jitsum(prob_z[:mid_z])+prob_z[mid_z]/2,jitsum(prob_z[mid_z+1:])+prob_z[mid_z]/2
        
    # prob xp1
    prob_dx_under_z=dx_under_z(k,dx_vec,zs,delta_t,var)
    prob_dx_under_x=dx_under_x(prob_dx_under_z,prob_z)

    Rh_left=prob_left*a-rho*tw
    Rh_right=prob_right*ana-rho*tw
    Rh_last=np.vstack((Rh_left[:,-1],Rh_right[:,-1],-np.ones_like(EV2c_vec),EV2c_vec))

    V=np.zeros((num_states,T))
    D=np.empty_like(V)
    EVnext=np.zeros_like(V)
    EVnext[:,-1]=-np.inf

    V[:,T-1],D[:,T-1]=max_(Rh_last)
    assert(not (np.isnan(V)).any())
    midz=175

    for iT in range(T-2,-1,-1):
        EVnext[:,iT]=expectation(V,prob_dx_under_x,xp1_state,iT)
        assert(np.abs(EVnext[:midz,iT].sum()-EVnext[midz+1:,iT].sum())<0.001)
        Rh=np.vstack((Rh_left[:,iT],Rh_right[:,iT]))
        assert(np.abs(Rh_left[:midz,iT].sum()-Rh_right[midz+1:,iT].sum())<0.001)
        assert(np.abs(Rh_right[:midz,iT].sum()-Rh_left[midz+1:,iT].sum())<0.001)
        zz=np.vstack((
            Rh,
            np.atleast_2d(EVnext[:,iT])-(rho+c)*delta_t,
            np.atleast_2d(EV2c_vec)
            ))
        # assert(np.sum(np.nan_to_num(zz[:,:midz]))-np.sum(np.nan_to_num(zz[:,midz+1:]))<0.00001)
        V[:,iT],D[:,iT]=max_(zz)
        assert(np.sum(V[:midz,iT])-np.sum(V[midz+1:,iT])<0.00001)
        assert(np.sum(D[:midz,iT])-np.sum(D[midz+1:,iT])<0.00001)
        # if iT==160:assert(1==0)
        assert(not (np.isnan(V)).any())
    V0=V[mid_state,0]
    return V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho


# ~ no accumulation


@njit(fastmath=True)
def big_Z_noaccu(k,zs,ps,var,x_grids,delta_t):
    prob_z=np.zeros((zs.shape[0],x_grids.shape[0],x_grids.shape[1]))
    
    C=2*delta_t*var
    x2_grids=2*x_grids
    kdt=k*delta_t
        
    for r in prange(zs.shape[0]):
        tmp=np.zeros_like(prob_z[r])
        for i in prange(zs.shape[0]):
            if i!=r:
                tmp=tmp+ps[i]/ps[r]*np.exp(kdt*(zs[i]-zs[r])*(x2_grids-kdt*(zs[i]+zs[r]))/C)
        prob_z[r]=1/(1+tmp)
    # assert(not (prob_z==0).any())
    return prob_z



@njit
def d_prob_noaccu(k,a,c,rho,tw,tu,EV2,zs,ps,sigma,num_states,delta_t,xbar=np.nan,stop_val=2,mapping_range=3):
    """discrete prior optimal policy model without accumulation process
    inspired by Stine et.al. Differentiating between integration andnon-integration strategies in perceptualdecision making, 2020

    Parameters
    ----------
    k : float
        drift rate
    a : float
        reward on left or right choice
    c : float
        cost per time unit
    rho : float
        average reward
    tw : float
        time for waiting
    tu : float 
        time for switching to second stage
    EV2 : float
        expected reward of stage 2
    zs : ndarray[float]
        coherence strength, zs is supposed to be symmetric
    ps : ndarray[float]
        probability
    sigma : float
        standard deviation of evidence
    num_states : int
        number of states on x(evidence)
    delta_t : float
        unit time interval
    xbar : float
        maximum x value
    stop_val : int, optional
        end of the time, by default 2
    mapping_range : int, optional
        times of sigma on x, by default 3

    Returns
    -------
        V0 : float
            expected value at the beginning of stage 1
        V : ndarray[float]
            value function under (t,x)-space
        D : ndarray[float]
            decision made in (t,x)-space
            0- left
            1- right
            2- waiting for the next evidence
            3- uncertain response
        EVnext : ndarray[float]
            expected value function under (t,x)-space
        rho : float
            average reward
    """
    
    tbar=np.floor(stop_val/delta_t)
    var=sigma**2
    t_vec=np.arange(delta_t,tbar*delta_t+1e-10,delta_t)

    T=len(t_vec)

    z_max=np.max(zs)
    z_min=np.min(zs)
    mid_state=int(np.floor(num_states/2))
    mid_z=int(np.floor(zs.shape[0]/2))

    sigma_range=mapping_range*sigma
    
    upper=k*z_max+3*sigma_range if np.isnan(xbar) else xbar 
    lower=-upper

    dx=(upper-lower)/(num_states-1)

    dx_vec=np.concatenate((
        np.flip(np.arange(0,-sigma_range,-dx))[:-1],
        np.arange(0,sigma_range,dx)))


    x_vec=np.arange(lower,upper+1e-10,dx)
    x_vec0=np.zeros_like(x_vec)

    EV2c=(EV2-rho*tu)
    EV2c_vec=EV2c*np.ones(num_states)
    xp1_mat=jitaddouter(x_vec0,dx_vec)
    xp1_state=findnearest(xp1_mat,x_vec) #(x,dx)

    x_grids,t_grids=jitmeshgrid(x_vec,t_vec)

    # prob preparation
    prob_z=big_Z_noaccu(k,zs,ps,var,x_grids,delta_t)
    # prob left and right
    if zs.shape[0]%2==0: # even
        prob_left,prob_right=jitsum(prob_z[:mid_z]),jitsum(prob_z[mid_z:])
    else: # odd
        prob_left,prob_right=jitsum(prob_z[:mid_z])+prob_z[mid_z]/2,jitsum(prob_z[mid_z+1:])+prob_z[mid_z]/2
        
    # prob xp1
    prob_dx_under_z=dx_under_z(k,dx_vec,zs,delta_t,var)
    prob_dx_under_x=dx_under_x(prob_dx_under_z,prob_z)


    Rh_left=prob_left*a-rho*tw
    Rh_right=prob_right*a-rho*tw
    Rh_last=np.vstack((Rh_left[:,-1],Rh_right[:,-1],EV2c_vec))

    V=np.zeros((num_states,T))
    D=np.empty_like(V)
    EVnext=np.zeros_like(V)
    EVnext[:,-1]=-np.inf

    V[:,T-1],D[:,T-1]=max_(Rh_last)
    assert(not (np.isnan(V)).any())

    for iT in range(T-2,-1,-1):
        EVnext[:,iT]=expectation(V,prob_dx_under_x,xp1_state,iT)
        Rh=np.vstack((Rh_left[:,iT],Rh_right[:,iT]))
        V[:,iT],D[:,iT]=max_(np.vstack((
            Rh,
            np.atleast_2d(EVnext[:,iT])-(rho+c)*delta_t,
            np.atleast_2d(EV2c_vec)
            )))
        assert(not (np.isnan(V)).any())
    V0=V[mid_state,0]
    
    return V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho




if __name__=='__main__':
    delta_t=0.02
    sigma=1
    num_states=351
    
    k=12.263363363424752
    a=11.665180127481781
    c=1
    rho=0
    tw1=1
    tw2=1
    tu=0.1
    zs=aa([-0.26,-0.16,-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0,\
        0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26])
    #zs=aa([-0.26,0.26])
    ps=aa([1.,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1])
    ps/=ps.sum()
    EV2=8.567450754173489
    
    V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho=d_prob(k,a,c,rho,tw2,tu,EV2,zs,ps,sigma,num_states,delta_t,ana=a,xbar=5.,stop_val=10,mapping_range=3)

    
    np.save('test_data/D.npy',D)
    # plt.imshow(D)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(V)
    # plt.colorbar()
    # plt.show()
    # ...