# generate different parameters for the model


# important for correctly savefig
import os
import numpy as np
aa=np.asarray
from py_model.Procedure import *
from py_model.PlentyResults import *
from py_model.Aux_ParametersStructure import *
import h5py
import pyperplot as ppp


subj="sall"
h5file=h5py.File(f"res/{subj}/bps.jld","r")
k,R,EV2all,nondt,frus=h5file['best_params'][()]
sims_folder='./sim_data/'

#%%
fns=['sim_a_f','sim_k_f','sim_k_a','sim_k_a_f_cross']
norm=lambda x:x/x.sum()
uniparams={"c":1,
    "rho":0, # if rho is 0, tu is useless
    "tw1":1,
    "tw2":1,
    "tu":0.1,
    "zs":aa([-0.26,-0.16,-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26]),
    "ps":norm(aa([1.,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1])),
    "zs2":aa([-0.26,0.26]),
    "ps2":aa([0.5,0.5]),
    "sigma":1.,
    "num_states":351,
    "delta_t":0.02,
    "stop_val":10,
    "stop_cutoff":0,
    "xbar":5}


tbar=np.floor(uniparams['stop_val']/uniparams['delta_t'])
t_vec=np.arange(uniparams['delta_t'],tbar*uniparams['delta_t']+1e-10,uniparams['delta_t'])

k_set=k+aa([-4,-2,0.,2,4])
R_set=R+aa([-3,-1.5,0.,1.5,3])
frus_set=frus+aa([-0.5,-0.25,0,0.25,0.5]) 
for j in range(4):
    sims_folder=os.path.join(sims_folder,fns[j])
    gs=GSSimulation(synthesis_all,len(t_vec),sims_folder)

    if j==0: # k
        gs.set_iter_param(k=k_set)
        gs.set_fix_param(a=R,frus=frus,**uniparams)
    elif j==1: # a
        gs.set_iter_param(a=R_set)
        gs.set_fix_param(k=k,frus=frus,**uniparams)
    elif j==2: # f
        gs.set_iter_param(frus=frus_set)
        gs.set_fix_param(k=k,a=R,**uniparams)
    elif j==3: # k,a,f cross
        gs.set_iter_param(k=k_set,a=R_set,frus=frus_set)
        gs.set_fix_param(**uniparams)
    gs.run(cross=True)
