# transfer fitted parameters from OptimRDKup fitting into figures

# important for correctly savefig
from typing import NoReturn
import numpy as np
from tqdm import tqdm
from scipy import stats
aa=np.asarray
from py_model.DiModel import *
from py_model.Aux_DiModel import *
from py_model.Aux_ParametersStructure import MinMaxMeanDiff
from py_model.MarkovFit import markov_fit
import glob,h5py

# %% [markdown]
# # Preparation
def synthesis(k,a,c,rho,frus,tw1,tw2,zs,ps,zs2,ps2,sigma,xbar,xbar2,num_states,delta_t,stop_val,ana,mapping_range=3):
    EV22=-np.inf
    V02,V2,D2,EVnext2,Rh_left2,Rh_right2,EV2c2,prob_left2,rho2_=d_prob(k,a,c,rho,tw2,tu,EV22,zs2,ps2,sigma,num_states,delta_t,xbar2,stop_val,ana,mapping_range)
    
    V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_=d_prob(k,a,c,rho,tw1,tu,V02-frus,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana,mapping_range)
    return V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_,V02,V2,D2,EVnext2,Rh_left2,Rh_right2,EV2c2,prob_left2,rho2_
# %%
def transfer_bounds_to_rt(k,zs,sigma,a_upper,a_lower,a_inner_upper,a_inner_lower,delta_t,xbar,num_states,stop_val,runmed_width,runmed_width_inner):
    g_inners=[]
    g_uppers=[]
    g_lowers=[]

    for i,z in enumerate(zs):
        g_upper,g_lower,g_inner,t_vec,ret_num_tr_states=markov_fit(
            ou_drift=k*z,ou_leak=0.,ou_var=sigma**2,ou_init=0.,
            a_upper=a_upper,a_lower=a_lower,
            a_inner_upper=a_inner_upper,a_inner_lower=a_inner_lower,
            delta_t=delta_t,
            covered_range=aa([-xbar,xbar]),num_tr_states=num_states,stop_val=stop_val,runmed_width=runmed_width,runmed_width_inner=runmed_width_inner)
        g_uppers.append(g_upper) 
        g_lowers.append(g_lower)
        g_inners.append(g_inner)

    return [aa(tmp) for tmp in [g_uppers,g_lowers,g_inners]]

# %%
def gen_dist(k,zs,sigma,
    a_upper,a_lower,a_inner_upper,a_inner_lower,
    a_upper2,a_lower2,a_inner_upper2,a_inner_lower2,
    a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std,
    delta_t,xbar,num_states,stop_val,runmed_width,runmed_width_inner):
    
    g_uppers,g_lowers,g_inners=transfer_bounds_to_rt(
        k,zs,sigma,a_upper,a_lower,a_inner_upper,a_inner_lower,delta_t,xbar,num_states,stop_val,runmed_width,runmed_width_inner)
    g_uppers2,g_lowers2,g_inners2=transfer_bounds_to_rt(
        k,zs2,sigma,a_upper2,a_lower2,a_inner_upper2,a_inner_lower2,delta_t,xbar,num_states,stop_val,runmed_width,runmed_width_inner)
    g_uppers_std,g_lowers_std,_=transfer_bounds_to_rt(
        k,zs,sigma,a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std,delta_t,xbar,num_states,stop_val,runmed_width,runmed_width_inner)
    
    return g_uppers,g_lowers,g_inners,g_uppers2,g_lowers2,g_inners2,g_uppers_std,g_lowers_std
#%%
delta_t=0.02 # fixed
sigma=1 # fixed
var=sigma**2
num_states=351 # fixed

stop_val=10. # stop_val (fixed)
mapping_range=3

tbar=np.floor(stop_val/delta_t)
t_vec=np.arange(delta_t,tbar*delta_t+1e-10,delta_t)


rho=0

# times
tw1=1 # fixed
tw2=1 # fixed
tu=0.1 # fixed

# Coherence and probability of stage1
zs=aa([-0.26,-0.16,-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26])
ps=aa([1.,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1])
ps/=ps.sum()
mid_z=int(np.floor(len(zs)//2))

# Coherence and probability of stage2
zs2=aa([-0.26,0.26])
ps2=aa([0.5,0.5])
EV22=-1
mid_z2=int(np.floor(len(zs2)//2))


# filter params
runmed_width=7
runmed_width_inner=7

# x range
xbar=5.;xbar2=5.

# %%#########################################################################
subjs=["s01","s02","s03","s06","s08","s09","s10","s12","s13","s15","sall"]
#subjs=["s01","s02","s03","s06","s08","s09","s10","s12","s13","s15"]
cond_name=""
param6model=False
following_oa=False
res_folder="res_pacc"
# %%#########################################################################
# %%

for subj in tqdm(subjs): # for each subject

    # determine the parameters
    best_file=f"./{res_folder}/{subj}/bps.jld"
    best_file_std=f"./{res_folder}/{subj}/bps.jld" if subj=='sall' or following_oa\
         else f"./{res_folder}/{subj}/bps_std.jld"

    # transfer parameters to numpy
    grdata_save={"subject":subj}
    save_file_path=f"../connect/data/{subj}{cond_name}.npy"

    c=1
    c_std=1
    # oa condition
    h5file=h5py.File(best_file,"r")
    if param6model:
        k,R,_,nondt,T0sigma,frus=h5file['best_params'][()]
        print("---OA---",k,R,nondt,T0sigma,frus)
    else:
        k,R,_,nondt,frus=h5file['best_params'][()]
        print("---OA---",k,R,nondt,frus)
    
    a=R*np.ones_like(t_vec)
    ana=a.copy()

    ## standard condition
    
    h5file_std=h5py.File(best_file_std,"r")
    
    if subj=='sall' or following_oa:
        if param6model:
            k_std,R_std,_,nondt_std,T0sigma_std,_=h5file_std['best_params'][()]
        else:
            k_std,R_std,_,nondt_std,_=h5file_std['best_params'][()]
    else:
        if param6model:
            k_std,R_std,nondt_std,T0sigma_std=h5file_std['best_params'][()]
        else:
            k_std,R_std,nondt_std=h5file_std['best_params'][()]
    if param6model:
        print("---STD---",k_std,R_std,nondt_std,T0sigma)
        assert(k==k_std and nondt==nondt_std and T0sigma==T0sigma_std)
    else:
        print("---STD---",k_std,R_std,nondt_std)
        assert(k==k_std and nondt==nondt_std)

    

    a_std=R_std*np.ones_like(t_vec)
    ana_std=R_std.copy()


    # %% CUE condition
    V0,V,D,EVnext,Rh_left,Rh_right,EV2c,prob_left,rho_,V02,V2,D2,EVnext2,Rh_left2,Rh_right2,EV2c2,prob_left2,rho2_=\
        synthesis(k,a,c,rho,frus,tw1,tw2,zs,ps,zs2,ps2,sigma,xbar,xbar2,num_states,delta_t,stop_val,ana,mapping_range=3)

    # %% STD condition
    EV2_std=-np.inf
    V0_std,V_std,D_std,EVnext_std,Rh_left_std,Rh_right_std,EV2c_std,prob_left_std,rho_std=d_prob(
        k_std,a_std,c_std,rho,tw1,tu,EV2_std,zs,ps,sigma,num_states,delta_t,xbar,stop_val,ana_std,mapping_range)

    # %%
    a_upper,a_lower,a_inner_upper,a_inner_lower=transfer_D_to_bounds(D)
    a_upper2,a_lower2,a_inner_upper2,a_inner_lower2=transfer_D_to_bounds(D2)
    a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std=transfer_D_to_bounds(D_std)

    # %% reaction time distributions
    g_uppers,g_lowers,g_inners,g_uppers2,g_lowers2,g_inners2,g_uppers_std,g_lowers_std=\
            gen_dist(k,zs,sigma,
                a_upper,a_lower,a_inner_upper,a_inner_lower,
                a_upper2,a_lower2,a_inner_upper2,a_inner_lower2,
                a_upper_std,a_lower_std,a_inner_upper_std,a_inner_lower_std,
                delta_t,xbar,num_states,stop_val,runmed_width,runmed_width_inner)

    grdata_save["area"]={
        "D":D.copy(),
        "D_std":D_std.copy(),
        "V":V.copy(),
        "V_std":V_std.copy(),
        "EVnext":V.copy(),
        "EVnext_std":V_std.copy(),
        "Rh_left":Rh_left.copy(),
        "Rh_right":Rh_right.copy(),
        "Rh_left_std":Rh_left_std.copy(),
        "Rh_right_std":Rh_right_std.copy(),
        'xlabel':'Time',
        'ylabel':'Evidence(x)'
    }
    
    
    # %%

    # %% mean response time and its variance


    if param6model:
        kernel=stats.norm(nondt,T0sigma).pdf(np.arange(0,nondt+3*T0sigma,delta_t))
        kernel/=kernel.sum()
        
        ng_inners=np.zeros_like(g_inners)
        ng_lowers=np.zeros_like(g_lowers)
        ng_uppers=np.zeros_like(g_uppers)
        for i in range(len(zs)):
            ng_inners[i]=np.convolve(g_inners[i],kernel,'full')[:len(g_inners[i])]
            ng_lowers[i]=np.convolve(g_lowers[i],kernel,'full')[:len(g_lowers[i])]
            ng_uppers[i]=np.convolve(g_uppers[i],kernel,'full')[:len(g_uppers[i])]


        ng_lowers_std=np.zeros_like(g_lowers_std)
        ng_uppers_std=np.zeros_like(g_uppers_std)
        for i in range(len(zs)):
            ng_lowers_std[i]=np.convolve(g_lowers_std[i],kernel,'full')[:len(g_lowers_std[i])]
            ng_uppers_std[i]=np.convolve(g_uppers_std[i],kernel,'full')[:len(g_uppers_std[i])]

        # cue
        g_inners_mu,g_inners_sigma=dist_mean_std(t_vec,ng_inners,zs)
        g_ul_mu,g_ul_sigma=dist_mean_std(t_vec,ng_lowers+ng_uppers,zs)

        # std
        g_ul_std_mu,g_ul_std_sigma=dist_mean_std(t_vec,ng_lowers_std+ng_uppers_std,zs)

        # %% uncertainty proportion

        prob_upper=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,ng_uppers)
        prob_lower=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,ng_lowers)
        prob_inner=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,ng_inners)
        up_prop=prob_inner/(prob_upper+prob_inner+prob_lower)


    else:
        # cue
        g_inners_mu,g_inners_sigma=dist_mean_std(t_vec,g_inners,zs)
        g_ul_mu,g_ul_sigma=dist_mean_std(t_vec,g_lowers+g_uppers,zs)
        g_ul_mu2,g_ul_sigma2=dist_mean_std(t_vec,g_lowers2+g_uppers2,zs2)

        # std
        g_ul_std_mu,g_ul_std_sigma=dist_mean_std(t_vec,g_lowers_std+g_uppers_std,zs)
    
        # %% uncertainty proportion

        prob_upper=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_uppers)
        prob_lower=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_lowers)
        prob_inner=np.apply_along_axis(lambda x:np.sum(x)*delta_t,1,g_inners)
        up_prop=prob_inner/(prob_upper+prob_inner+prob_lower)


    # correct rate

    if param6model:
        crc_std,crc_opt_lr,crc_opt_lr2=cr_coh(t_vec,ng_uppers,ng_lowers,g_uppers2,g_lowers2,ng_uppers_std,ng_lowers_std,zs,mid_z,phymetric=False)
    else:
        crc_std,crc_opt_lr,crc_opt_lr2=cr_coh(t_vec,g_uppers,g_lowers,g_uppers2,g_lowers2,g_uppers_std,g_lowers_std,zs,mid_z,phymetric=False)

    half=mid_z
    crc_std=crc_std[half:]
    crc_opt_lr=crc_opt_lr[half:]
    crc_opt_lr2=crc_opt_lr2[half:]

    grdata_save["accuracy"]={
        "x":zs[half:].copy(),
        "y_no-up lr":crc_std.copy(),
        "y_opt-up lr":crc_opt_lr.copy(),
        "y_opt-up 2nd":crc_opt_lr2.copy(),
        'xlabel':'|Coherence|',
        'ylabel':'Accuracy'
    }

    # %%
    N=25
    alpha=0.5


    if param6model:
        grdata_save["rt"]={
            "x":zs.copy(),
            "y_no-up lr":(g_ul_std_mu).copy(),
            "yerr_no-up lr":(g_ul_std_sigma/np.sqrt(0.5*N)).copy(),
            "y_opt-up lr":(g_ul_mu).copy(),
            "yerr_opt-up lr": (g_ul_sigma/np.sqrt((1-up_prop)*N)).copy(),
            "y_opt-up up":(g_inners_mu).copy(),
            "yerr_opt-up up":(g_inners_sigma/np.sqrt(up_prop*N)).copy(),
            'xlabel':'Cohernece',
            'ylabel':'Reaction time (s)'
        }

    else:
        grdata_save["rt"]={
            "x":zs.copy(),
            "y_no-up lr":(g_ul_std_mu+nondt_std).copy(),
            "yerr_no-up lr":(g_ul_std_sigma/np.sqrt(0.5*N)).copy(),
            "y_opt-up lr":(g_ul_mu+nondt).copy(),
            "yerr_opt-up lr": (g_ul_sigma/np.sqrt((1-up_prop)*N)).copy(),
            "y_opt-up up":(g_inners_mu+nondt).copy(),
            "yerr_opt-up up":(g_inners_sigma/np.sqrt(up_prop*N)).copy(),
            'xlabel':'Cohernece',
            'ylabel':'Reaction time (s)'
        }
        grdata_save["rt2"]={
            "x":zs2.copy(),
            "y_opt-up 2nd":(g_ul_mu2+nondt).copy(),
            'xlabel':'Cohernece',
            'ylabel':'Reaction time (s)'
        }
    # %%
    half=mid_z#mid_z

    grdata_save["ur_prop"]={
        "x":zs[half:].copy(),
        "y_opt-up":up_prop[half:].copy(),
        'xlabel':"|Cohernece|",
        'ylabel':"UR proportion"
    }

    # %% reaction time quantial and accuracy
    import pandas as pd

    def RTAccQuantile(g_uppers,g_lowers,g_uppers_std,g_lowers_std,pieces=5,right_side=True):
        len_zs=g_uppers.shape[0]
        if right_side:
            s=mid_z;e=len_zs
        else:
            s=0;e=mid_z+1
        pdf_opt=np.sum(ps[s:e].reshape(-1,1)*(g_uppers+g_lowers)[s:e],axis=0)
        pdf_std=np.sum(ps[s:e].reshape(-1,1)*(g_uppers_std+g_lowers_std)[s:e],axis=0)
        
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
            correct_opt=g_uppers[s:e,p1:p2].sum()/pdf_opt[p1:p2].sum()
            wrong_opt=g_lowers[s:e,p1:p2].sum()/pdf_opt[p1:p2].sum()
            correct_std=g_uppers_std[s:e,p1:p2].sum()/pdf_std[p1:p2].sum()
            wrong_std=g_lowers_std[s:e,p1:p2].sum()/pdf_std[p1:p2].sum()
            if right_side:
                cr_opt.append(correct_opt/(correct_opt+wrong_opt))
                cr_std.append(correct_std/(correct_std+wrong_std))
            else:
                cr_opt.append(wrong_opt/(correct_opt+wrong_opt))
                cr_std.append(wrong_std/(correct_std+wrong_std))
        return cr_opt,cr_std

    cr_opt,cr_std=RTAccQuantile(g_uppers,g_lowers,g_uppers_std,g_lowers_std)
    cr_opt_left,cr_std_left=RTAccQuantile(g_uppers,g_lowers,g_uppers_std,g_lowers_std,right_side=False)

    # %%
    grdata_save["acc_quantile"]={
        "x":np.arange(5),
        "y_no-up":cr_std.copy(),
        "y_opt-up lr":cr_opt.copy(),
        'xlabel':"Reaction time (s) quintile",
        'ylabel':"Accuracy"
    }

    pdf_opt=np.sum((g_uppers+g_lowers).T*ps,axis=1)
    pdf_ur=np.sum(g_inners.T*ps,axis=1)
    
    # %% single point
    grdata_save["point"]={
        "rt_opt-lr":np.sum(ps*g_ul_mu),
        "rt_opt-up":np.sum(ps*g_inners_mu),
        "rt_diff_opt-lr-up":np.sum(ps*g_ul_mu)-np.sum(ps*g_inners_mu),
        "rt_diff_opt-lr-up_norm":MinMaxMeanDiff(pdf_opt,pdf_ur), # normalized RT difference
        "acc_opt-lr":np.mean(crc_opt_lr),
        "acc_std":np.mean(crc_std),
        "acc_diff_opt-lr-std":np.mean(crc_opt_lr)-np.mean(crc_std),
        "UR_prop":np.sum(ps*up_prop)
    }
    grdata_save["rel_area"]={
        "rel_UR":np.sum(D==3)/np.sum((D==2) | (D==3))
    }

    # %% save the file
    np.save(save_file_path,grdata_save)

