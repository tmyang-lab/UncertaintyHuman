import numpy as np
import scipy.stats as stats
from .utils import runmed
from numba import njit
import math
aa=np.asarray

@njit
def jitpdf(x,mu,sigma):
    val=1/np.sqrt(2*np.pi)*np.exp(-((x-mu)/sigma)**2/2)
    return val/np.sum(val)
    

# Local functions
@njit
def val2state(val,covered_range,num_states):
    """
    The states are numbered from 0 to num_states-1. These are the 2 absorbing states.
    The transient states are numbered from 1 to num_states-2.
    """
    temp=(val-covered_range[0]+(covered_range[1]-covered_range[0])/(2*num_states))/(covered_range[1]-covered_range[0])*(num_states-2)
    if temp<.5:
        temp=0
    if temp>num_states-1.5:
        temp=num_states-1
    state=int(np.round(temp))

    return state

@njit
def state2val(state,covered_range,num_states):
    val=covered_range[0]+(state-.5)*(covered_range[1]-covered_range[0])/num_states
    return val

@njit
def incval2stateinc(inc_val,covered_range,num_states):
    state_inc=inc_val/(covered_range[1]-covered_range[0])*num_states
    return state_inc

@njit
def nearest_state(state,num_states):
    """
    inner bound to state
    """
    if state>num_states-1.5:state=num_states-1
    if state<0.5:state=0

    state=int(np.round(state))
    return state

@njit
def markov_fit(ou_drift,ou_leak,ou_var,ou_init,a_upper,a_lower,a_inner_upper,a_inner_lower,delta_t,covered_range,num_tr_states,stop_val,mapping_range=3,runmed_width=3,runmed_width_inner=3):
    """Markov chain approximation of the first passage time problem
    for a (time-variant) 1D Ornstein-Uhlenbeck(leaky = 0 for now) process with 2 (time-variant) boundaries and 1 (time-variant) inner boundaries. The calculation is based on a Random Walk with Gaussian Increments.

    Parameters
    ----------
    ou_drift : float
        is the drift of the OU process. 
    ou_leak : float
        defines the "leakiness" of the integrator. The deterministic part of the stochastic differential equation is given by ou_drift - ou_leak * current_value. A Wiener process can be studied by setting ou_leak to 0. It is always 0 for now.
    ou_var : float
        is the variance of the OU process. It can either be a constant or, for the time-variant case, the name of a function, which must return the variance when called with the time as the argument.
    ou_init : float
        is the initial value of the OU process. 
    a_upper : ndarray[float]
        defines the upper absorbing boundary. a_upper must return the location of the boundary when called with the time as the argument.
    a_lower : ndarray[float]
        defines the lower absorbing boundary. See a_upper for the format.
    a_inner_upper : ndarray[float]
        defines the uppeinner absorbing boundary. a_inner_upper must return the location of the boundary when called with the time as the argument.
    a_inner_lower : ndarray[float]
        defines the lower inner absorbing boundary. a_inner must return the location of the boundary when called with the time as the argument.
    delta_t : float
        is the temporal step size.
    covered_range : list[float] or  ndarray[float]
        defines the range, which should be covered by the transient states.
        It has to be a 2 element vector. Make sure that you never define boundaries outside the specified range! (In this case the algorithm doesn't solve the problem you are interested in.) In the case of constant boundaries the best results are obtained when the boundaries are identical to covered_range(1) and covered_range(2).
    num_tr_states : int
        defines the number of transient states used for the discrete approximation of the problem. When using a symmetrical range you should specify an odd number of states for an unbiased representation of 0. When passing 0 a built-in mechanism is used for choosing an optimal number of states (based on a heuristic method).
    stop_val : float
        defines the time or the error, which determines when the algorithm will stop.
    mapping_range : int, optional
        defines how long the vectors used for constructing the transition matrix are and therefore how sparse the transition matrix will be. The mapped range is MEAN +/- mapping_range * SD., by default 3
    runmed_width : int, optional
        defines the width of a running median filter applied to the output in case of time-variant boundaries to remove spikes. It has to be an odd number., by default 3
    runmed_width_inner : int, optional
         defines the width of a running median filter applied to the output in case of time-variant boundaries to remove spikes. It has to be an odd number., by default 3

    Returns
    -------
    g_upper : ndarray[float] 
        is the first passage time density for the upper boundary multiplied by the probability of hitting the upper boundary first, evaluated at the times given in t_vec. If ou_init is a vector, g_upper is a matrix. Each row represents the FPT density for one of the initial values.
    g_lower : ndarray[float]
        is the first passage time density for the lower boundary multiplied by the probability of hitting the lower boundary first, evaluated at the times given in t_vec. If ou_init is a vector, g_lower is a matrix. Each row represents the FPT density for one of the initial values.
    g_inner : ndarray[float]
        is the first passage time density for the inner boundary multiplied by the probability of hitting the inner boundary first, evaluated at the times given in t_vec. If ou_init is a vector, g_lower is a matrix. Each row represents the FPT density for one of the initial values.
    t_vec : ndarray[float]
        is the time indicator.
    ret_num_tr_states : int
        is the number of transient states. This is for the case when you use the built-in mechanism for choosing an optimal number of states and you would like to know what spatial resolution was chosen.
    """    

    # state number identification
    num_states=num_tr_states 
    extra_num_states=num_states+1 # + upper absorbing, lower absorbing, inner absorbing bound
    inner_state=num_states
    ret_num_tr_states=num_tr_states

    # initalization
    last_a_lower=np.nan
    last_a_upper=np.nan

    drift_cur=incval2stateinc(ou_drift*delta_t,covered_range,num_states)
    sd_cur=incval2stateinc(np.sqrt(ou_var*delta_t),covered_range,num_states)
    ou_init_state=val2state(ou_init,covered_range,num_states)

    current_mat=np.diag(np.ones(extra_num_states))

    # Loop
    kbar=np.floor(stop_val/delta_t)
    t_vec=np.arange(delta_t,kbar*delta_t+1e-10,delta_t)
    g_lower=np.zeros(t_vec.shape,dtype=np.float64)
    g_upper=np.zeros(t_vec.shape,dtype=np.float64)
    g_inner=np.zeros(t_vec.shape,dtype=np.float64)

    t=0
    a_lower_cur=a_lower[0]
    a_upper_cur=a_upper[0]


    inner_area=False if a_inner_upper[0]==-1 else True
    # assert(not inner_area)
    if inner_area:
        a_inner_upper_cur,a_inner_lower_cur=a_inner_upper[0],a_inner_lower[0]
    last_a_lower=a_lower_cur
    last_a_upper=a_upper_cur
    if inner_area:
        last_a_inner_lower_cur=a_inner_lower_cur
        last_a_inner_upper_cur=a_inner_upper_cur
 
    m_cur=drift_cur
    
    lower_bound=int(np.floor(m_cur-mapping_range*sd_cur))# relative state moving
    upper_bound=int(np.ceil(m_cur+mapping_range*sd_cur))
    
    x=np.arange(lower_bound,upper_bound+1e-5,1)
    y=jitpdf(x,m_cur,sd_cur)

    for i in range(len(t_vec)):
        
        t=t_vec[i]
        a_lower_cur=a_lower[i]
        a_upper_cur=a_upper[i]

        last_inner_area=inner_area
        inner_area=False if a_inner_upper[i]==-1 else True
        if inner_area:
            a_inner_upper_cur,a_inner_lower_cur=int(a_inner_upper[i]),int(a_inner_lower[i])

        # construct new transition matrix     
        helpm=np.zeros((extra_num_states,extra_num_states)) # transition matrix
        helpm[inner_state,inner_state]=1 # inner state
        
    
        for row in range(0,num_states): # This is the "from" state.
            
            if row<=last_a_lower:
                helpm[row,a_lower_cur]=1
                continue
            if row>=last_a_upper:
                helpm[row,a_upper_cur]=1
                continue
            if inner_area and last_inner_area==inner_area and (row>=last_a_inner_lower_cur and row<=last_a_inner_upper_cur):
                helpm[row,inner_state]=1.
                continue

            index=-row # independent of row

            # inner_area=False
            
            for col in range(0,num_states): # This is the "to" state (+1).

                if col==a_lower_cur: # first col is special (lower absorbing boundary)
                    if row<=last_a_lower:
                        helpm[row,col]=1
                    elif row>=last_a_upper: # source state above (or part of) upper boundary?
                        ...# helpm[row,col]=0 # is absorbed by UPPER boundary
                    elif -row+a_lower_cur>=lower_bound: # entry necessary ?
                        if -row+a_lower_cur>upper_bound:
                            helpm[row,col]=1
                        else:
                            helpm[row,col]=np.sum(y[:-row+a_lower_cur-lower_bound+1])

                elif col==a_upper_cur: # last col is special (upper absorbing boundary)
                    if row<=last_a_lower:
                        ...
                    elif row>=last_a_upper: # source state above (or part of) upper boundary?
                        helpm[row,col]=1 # is absorbed by upper boundary
                    elif -row+a_upper_cur<=upper_bound: # entry necessary ?
                        if -row+a_upper_cur<lower_bound:
                            helpm[row,col]=1
                        else:
                            helpm[row,col]=np.sum(y[-row+a_upper_cur-lower_bound:])
                
                else: # standard col
                    if row<=last_a_lower:
                        ...
                    elif row>=last_a_upper:
                        ...
                    elif col<a_lower_cur or col>a_upper_cur:
                        ...
                    elif index>=lower_bound and index<=upper_bound:
                        helpm[row,col]=y[index-lower_bound]
                index+=1

            if inner_area:
                
                if last_inner_area==inner_area:
                    if row>last_a_inner_upper_cur: # problem here
                        helpm[row,inner_state]=np.sum(helpm[row,:a_inner_upper_cur+1])
                        helpm[row,:a_inner_upper_cur+1]=0
                    elif row<last_a_inner_lower_cur:
                        helpm[row,inner_state]=np.sum(helpm[row,a_inner_lower_cur:-1])
                        helpm[row,a_inner_lower_cur:-1]=0
                else:
                    helpm[row,inner_state]=np.sum(helpm[row,a_inner_lower_cur:a_inner_upper_cur+1])
                    helpm[row,a_inner_lower_cur:a_inner_upper_cur+1]=0

        last_a_lower=a_lower_cur
        last_a_upper=a_upper_cur

        if inner_area:
            last_a_inner_lower_cur=a_inner_lower_cur
            last_a_inner_upper_cur=a_inner_upper_cur
        else:
            last_a_inner_lower_cur=-1
            last_a_inner_upper_cur=num_states+1
        

        p=current_mat[ou_init_state].copy()
        p[:last_a_lower+1]=0;p[last_a_upper:num_states]=0
        
        if inner_area and last_inner_area==inner_area:
            p[inner_state]=0
        
        if inner_area:
            g_inner[i]=np.sum(p*helpm[:,inner_state])/delta_t
        if a_upper_cur!=-1:
            g_upper[i]=np.sum(p*helpm[:,a_upper_cur])/delta_t
            g_lower[i]=np.sum(p*helpm[:,a_lower_cur])/delta_t
        else:
            g_upper[i]=0
            g_lower[i]=0
        
        current_mat=current_mat@helpm

    g_inner=runmed(g_inner,runmed_width_inner,1,0)
    g_upper=runmed(g_upper,runmed_width,1,0)
    g_lower=runmed(g_lower,runmed_width,1,0)



    return g_upper,g_lower,g_inner,t_vec,ret_num_tr_states



if __name__=="__main__":
    
    # import matplotlib.pyplot as plt
    # from utils import plot_dist
    import numpy as np
    a_upper,a_lower,a_inner_upper,a_inner_lower=np.load('test_data/boundsD.npy')

    delta_t=0.01
    sigma=1.
    num_states=301
    k=0
    xbar=2.2


    g_upper,g_lower,g_inner,t_vec,ret_num_tr_states=markov_fit(
        ou_drift=k,ou_leak=0.,ou_var=sigma**2,ou_init=0.,
        a_upper=a_upper,a_lower=a_lower,
        a_inner_upper=a_inner_upper,a_inner_lower=a_inner_lower,
        delta_t=delta_t,
        covered_range=[-xbar,xbar],num_tr_states=num_states,stop_val=4.)
    print("num_tr_states",ret_num_tr_states)
    
    # plot_dist(t_vec,g_upper,g_lower,g_inner_upper=None,g_inner_lower=None,g_inner=g_inner,save="markov_%d.png"%num_states)
    # plt.show()
    

    

