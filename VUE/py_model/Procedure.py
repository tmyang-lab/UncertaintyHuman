from .DiModel import *
from .Aux_DiModel import *
from .MarkovFit import markov_fit
from numba import jit,njit
import numpy as np
aa=np.asarray



def bar_search(k,zs,ps,sigma,var,stop_val,mid_z,mapping_range=3):
    @njit
    def func(x):
        return big_Z_bound(k,zs,ps,var,x,stop_val,mid_z)
    return binary_search(func,0,5*(k*np.max(zs)*stop_val+mapping_range*sigma),0.2)

def dist_mean_std(t_vec,dist,z_side):
    g_mu=[];g_sigma=[]
    for i,z in enumerate(z_side):
        g_mu.append(np.sum(t_vec*dist[i])/dist[i].sum())
        g_sigma.append(np.sum(dist[i]/dist[i].sum()*(t_vec-g_mu[-1])**2))
    return aa(g_mu),aa(g_sigma)


def synthesis_opt(k,a,c,rho,frus,tw1,tw2,tu,zs,ps,zs2,ps2,sigma,xbar,xbar2,num_states,delta_t,stop_val,ana=np.nan,stop_cutoff=0,mapping_range=3):
    EV22=-np.inf
    V02,V2,D2,EVnext2,Rh_left2,Rh_right2,EV2c2,prob_left2,rho2_=d_prob(k,a,c,rho,tw2,tu,EV22,zs2,ps2,sigma,num_states,delta_t,xbar2,stop_val,ana,mapping_range)
    V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_=d_prob(k,a,c,rho,tw1,tu,V02-frus,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana,mapping_range)

    # cutoff
    
    idx_end=int((stop_val-stop_cutoff)/delta_t)

    V=V[:,:idx_end]
    D=D[:,:idx_end]
    EVnext=EVnext[:,:idx_end]
    Rh_left=Rh_left[:,:idx_end]
    Rh_right=Rh_right[:,:idx_end]
    prob_left=prob_left[:,:idx_end]

    V2=V2[:,:idx_end]
    D2=D2[:,:idx_end]
    EVnext2=EVnext2[:,:idx_end]
    Rh_left2=Rh_left2[:,:idx_end]
    Rh_right2=Rh_right2[:,:idx_end]
    prob_left2=prob_left2[:,:idx_end]

    return V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_,V02,V2,D2,EVnext2,Rh_left2,Rh_right2,EV2c2,prob_left2,rho2_


def sythesis_std(k,a,c,rho,tw1,tu,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana=np.nan,stop_cutoff=0,EV2_std=-np.inf,mapping_range=3):
    V0_std,V_std,D_std,EVnext_std,Rh_left_std,Rh_right_std,EV2c_std,prob_left_std,rho_std=d_prob(k,a,c,rho,tw1,tu,EV2_std,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana,mapping_range)

    #cutoff

    idx_end=int((stop_val-stop_cutoff)/delta_t)

    V_std=V_std[:,:idx_end]
    D_std=D_std[:,:idx_end]
    EVnext_std=EVnext_std[:,:idx_end]
    Rh_left_std=Rh_left_std[:,:idx_end]
    Rh_right_std=Rh_right_std[:,:idx_end]
    prob_left_std=prob_left_std[:,:idx_end]

    return V0_std,V_std,D_std,EVnext_std,Rh_left_std,Rh_right_std,EV2c_std,prob_left_std,rho_std


def synthesis_all(k,a,c,rho,frus,tw1,tw2,tu,zs,ps,zs2,ps2,sigma,num_states,delta_t,stop_val,xbar=np.nan,ana=np.nan,stop_cutoff=0,mapping_range=3,runmed_width=3,runmed_width_inner=3):
    """
    k,a,c affect the transition probabilities
    """
    var=sigma**2
    mid_z=int(np.floor(len(zs)//2))
    if np.isnan(xbar):
        xbar=bar_search(k,zs,ps,sigma,var,stop_val,mid_z,mapping_range=mapping_range)
    xbar2=xbar

    print('model computing...')
    V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_,V02,V2,D2,EVnext2,Rh_left2,Rh_right2,EV2c2,prob_left2,rho2_=synthesis_opt(k,a,c,rho,frus,tw1,tw2,tu,zs,ps,zs2,ps2,sigma,xbar,xbar2,num_states,delta_t,stop_val,ana,stop_cutoff=stop_cutoff,mapping_range=mapping_range)
    V0_std,V_std,D_std,EVnext_std,Rh_left_std,Rh_right_std,EV2c_std,prob_left_std,rho_std=sythesis_std(k,a,c,rho,tw1,tu,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana,stop_cutoff=stop_cutoff,EV2_std=-np.inf,mapping_range=mapping_range)

    # transfer decision area to bound and rt probabilities
    print('bound transferring...')
    a_upper,a_lower,a_inner_upper,a_inner_lower=transfer_D_to_bounds(D)
    a_upper2,a_lower2,a_inner_upper2,a_inner_lower2=transfer_D_to_bounds(D2)
    a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std=transfer_D_to_bounds(D_std)

    def transfer_bounds_to_rt(a_upper,a_lower,a_inner_upper,a_inner_lower):
        g_inners=[]
        g_uppers=[]
        g_lowers=[]

        z_side=zs # both side
        for i,z in enumerate(z_side):
            g_upper,g_lower,g_inner,t_vec,ret_num_tr_states=markov_fit(
                ou_drift=k*z,ou_leak=0.,ou_var=var,ou_init=0.,
                a_upper=a_upper,a_lower=a_lower,
                a_inner_upper=a_inner_upper,
                a_inner_lower=a_inner_lower,
                delta_t=delta_t,
                covered_range=aa([-xbar,xbar]),num_tr_states=num_states,stop_val=stop_val-stop_cutoff,runmed_width=runmed_width,runmed_width_inner=runmed_width_inner)
            g_uppers.append(g_upper) 
            g_lowers.append(g_lower)
            g_inners.append(g_inner)
        return tuple([aa(tmp) for tmp in [g_uppers,g_lowers,g_inners]]+[t_vec])

    # distribution functions
    print('creating distribution...')
    g_uppers,g_lowers,g_inners,t_vec=transfer_bounds_to_rt(a_upper,a_lower,a_inner_upper,a_inner_lower)
    g_uppers2,g_lowers2,g_inners2,_=transfer_bounds_to_rt(a_upper2,a_lower2,a_inner_upper2,a_inner_lower2)

    g_uppers_std,g_lowers_std,g_inners_std,_=transfer_bounds_to_rt(a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std)

    prob_upper=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_uppers)
    prob_lower=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_lowers)
    prob_inner=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_inners)
    up_prop=prob_inner/(prob_upper+prob_inner+prob_lower)

    # mean and variance of response time
    print('creating mean variance of distribution...')
    z_side=zs

    # UR
    g_inners_mu,g_inners_sigma=dist_mean_std(t_vec,g_inners,z_side)
    g_ul_mu,g_ul_sigma=dist_mean_std(t_vec,g_uppers+g_lowers,z_side)
    
    # std

    g_ul_std_mu,g_ul_std_sigma=dist_mean_std(t_vec,g_uppers_std+g_lowers_std,z_side)

    #
    print('RT quantiled...') 
    def RTAccQuantile(g_uppers,g_lowers,g_uppers_std,g_lowers_std,pieces=5):
    
        pdf_opt=np.sum(ps[mid_z:].reshape(-1,1)*(g_uppers+g_lowers)[mid_z:],axis=0)
        pdf_std=np.sum(ps[mid_z:].reshape(-1,1)*(g_uppers_std+g_lowers_std)[mid_z:],axis=0)
        
        pdf=pdf_opt+pdf_std
        
        cdf=np.add.accumulate(pdf)
        unit=cdf[-1]/pieces
        piece=1
        qs=[0];delta=np.inf
        for i in range(len(cdf)):
            z=cdf[i]-unit
            if z>0:
                qs.append(i)
                piece+=1
                if piece==pieces:break
                unit=piece/pieces*cdf[-1]
        qs.append(len(cdf))
        cr_opt=[]
        cr_std=[]
        for i,q in enumerate(qs[:-1]):
            p1,p2=qs[i],qs[i+1]
            correct_opt=g_uppers[mid_z:,p1:p2].sum()/pdf_opt[p1:p2].sum()
            wrong_opt=g_lowers[mid_z:,p1:p2].sum()/pdf_opt[p1:p2].sum()
            correct_std=g_uppers_std[mid_z:,p1:p2].sum()/pdf_std[p1:p2].sum()
            wrong_std=g_lowers_std[mid_z:,p1:p2].sum()/pdf_std[p1:p2].sum()
            cr_opt.append(correct_opt/(correct_opt+wrong_opt))
            cr_std.append(correct_std/(correct_std+wrong_std))
        return cr_opt,cr_std
    cr_opt,cr_std=RTAccQuantile(g_uppers,g_lowers,g_uppers_std,g_lowers_std)

    return V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_,V02,V2,D2,EVnext2,Rh_left2,Rh_right2,EV2c2,prob_left2,rho2_,V0_std,V_std,D_std,EVnext_std,Rh_left_std,Rh_right_std,EV2c_std,prob_left_std,rho_std,g_uppers,g_lowers,g_inners,g_uppers2,g_lowers2,g_inners2,g_uppers_std,g_lowers_std,g_inners_std,t_vec,prob_upper,prob_lower,prob_inner,up_prop,xbar,xbar2,g_ul_mu,g_ul_sigma,g_inners_mu,g_inners_sigma,g_ul_std_mu,g_ul_std_sigma,cr_opt,cr_std


def synthesis_opt_direct(k,a,c,rho,EV2all,tw1,tu,zs,ps,sigma,xbar,num_states,delta_t,stop_val,ana=np.nan,stop_cutoff=0,mapping_range=3):
    """
    k,a,c does not affect second stage estimation
    """
    V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_=d_prob(k,a,c,rho,tw1,tu,EV2all,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana,mapping_range)

    # cutoff    
    idx_end=int((stop_val-stop_cutoff)/delta_t)

    V=V[:,:idx_end]
    D=D[:,:idx_end]
    EVnext=EVnext[:,:idx_end]
    Rh_left=Rh_left[:,:idx_end]
    Rh_right=Rh_right[:,:idx_end]
    prob_left=prob_left[:,:idx_end]

    return V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_



def synthesis_all_direct(k,a,c,rho,EV2all,tw1,tu,zs,ps,sigma,num_states,delta_t,stop_val,xbar=np.nan,ana=np.nan,stop_cutoff=0,mapping_range=3,runmed_width=3,runmed_width_inner=3):
    """
    k,a,c does not affect second stage estimation
    """

    var=sigma**2
    mid_z=int(np.floor(len(zs)//2))
    if np.isnan(xbar):
        xbar=bar_search(k,zs,ps,sigma,var,stop_val,mid_z,mapping_range=mapping_range)

    print('(direct) model computing...')
    V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_=synthesis_opt_direct(k,a,c,rho,EV2all,tw1,tu,zs,ps,sigma,xbar,num_states,delta_t,stop_val,ana,stop_cutoff=0,mapping_range=3)
    V0_std,V_std,D_std,EVnext_std,Rh_left_std,Rh_right_std,EV2c_std,prob_left_std,rho_std=sythesis_std(k,a,c,rho,tw1,tu,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana,stop_cutoff=stop_cutoff,EV2_std=-np.inf,mapping_range=mapping_range)

    # transfer decision area to bound and rt probabilities
    print('(direct) bound transferring...')
    a_upper,a_lower,a_inner_upper,a_inner_lower=transfer_D_to_bounds(D)
    a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std=transfer_D_to_bounds(D_std)

    def transfer_bounds_to_rt(a_upper,a_lower,a_inner_upper,a_inner_lower):
        g_inners=[]
        g_uppers=[]
        g_lowers=[]

        z_side=zs # both side
        for i,z in enumerate(z_side):
            g_upper,g_lower,g_inner,t_vec,ret_num_tr_states=markov_fit(
                ou_drift=k*z,ou_leak=0.,ou_var=var,ou_init=0.,
                a_upper=a_upper,a_lower=a_lower,
                a_inner_upper=a_inner_upper,
                a_inner_lower=a_inner_lower,
                delta_t=delta_t,
                covered_range=aa([-xbar,xbar]),num_tr_states=num_states,stop_val=stop_val-stop_cutoff,runmed_width=runmed_width,runmed_width_inner=runmed_width_inner)
            g_uppers.append(g_upper) 
            g_lowers.append(g_lower)
            g_inners.append(g_inner)

        return tuple([aa(tmp) for tmp in [g_uppers,g_lowers,g_inners]]+[t_vec])

    # distribution functions
    print('(direct) creating distribution...')
    g_uppers,g_lowers,g_inners,t_vec=transfer_bounds_to_rt(a_upper,a_lower,a_inner_upper,a_inner_lower)

    g_uppers_std,g_lowers_std,g_inners_std,_=transfer_bounds_to_rt(a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std)

    prob_upper=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_uppers)
    prob_lower=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_lowers)
    prob_inner=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_inners)
    up_prop=prob_inner/(prob_upper+prob_inner+prob_lower)

    # mean and variance of response time
    print('(direct) creating mean variance of distribution...')
    z_side=zs



    # UR
    g_ul_mu,g_ul_sigma=dist_mean_std(t_vec,g_uppers+g_lowers,z_side)
    g_inners_mu,g_inners_sigma=dist_mean_std(t_vec,g_inners,z_side)

    # std
    g_ul_std_mu,g_ul_std_sigma=dist_mean_std(t_vec,g_lowers_std+g_uppers_std,z_side)
    
    #
    print('(direct) RT quantiled...') 
    def RTAccQuantile(g_uppers,g_lowers,g_uppers_std,g_lowers_std,pieces=5):
    
        pdf_opt=np.sum(ps[mid_z:].reshape(-1,1)*(g_uppers+g_lowers)[mid_z:],axis=0)
        pdf_std=np.sum(ps[mid_z:].reshape(-1,1)*(g_uppers_std+g_lowers_std)[mid_z:],axis=0)
        
        pdf=pdf_opt+pdf_std
        
        cdf=np.add.accumulate(pdf)
        unit=cdf[-1]/pieces
        piece=1
        qs=[0];delta=np.inf
        for i in range(len(cdf)):
            z=cdf[i]-unit
            if z>0:
                qs.append(i)
                piece+=1
                if piece==pieces:break
                unit=piece/pieces*cdf[-1]
        qs.append(len(cdf))
        cr_opt=[]
        cr_std=[]
        for i,q in enumerate(qs[:-1]):
            p1,p2=qs[i],qs[i+1]
            correct_opt=g_uppers[mid_z:,p1:p2].sum()/pdf_opt[p1:p2].sum()
            wrong_opt=g_lowers[mid_z:,p1:p2].sum()/pdf_opt[p1:p2].sum()
            correct_std=g_uppers_std[mid_z:,p1:p2].sum()/pdf_std[p1:p2].sum()
            wrong_std=g_lowers_std[mid_z:,p1:p2].sum()/pdf_std[p1:p2].sum()
            cr_opt.append(correct_opt/(correct_opt+wrong_opt))
            cr_std.append(correct_std/(correct_std+wrong_std))
        return cr_opt,cr_std

    cr_opt,cr_std=RTAccQuantile(g_uppers,g_lowers,g_uppers_std,g_lowers_std)

    return V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_,V0_std,V_std,D_std,EVnext_std,Rh_left_std,Rh_right_std,EV2c_std,prob_left_std,rho_std,g_uppers,g_lowers,g_inners,g_uppers_std,g_lowers_std,g_inners_std,t_vec,prob_upper,prob_lower,prob_inner,up_prop,xbar,g_ul_mu,g_ul_sigma,g_inners_mu,g_inners_sigma,g_ul_std_mu,g_ul_std_sigma,cr_opt,cr_std
