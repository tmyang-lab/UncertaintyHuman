
'''
Filename: /home/flumer/Documents/Project/OptimRDKup/model/PlentyResults.py
Path: /home/flumer/Documents/Project/OptimRDKup/model
Created Date: Wednesday, June 23rd 2021, 2:25:16 pm
Author: LI Xiaodong

To be generalized in the future
'''


import numpy as np
import inspect,itertools,os,glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyperplot as ppp
from tqdm import tqdm

class GSSimulation:
    """
    generate and save simulation
    """
    def __init__(self,func,len_t_vec=0,sim_folder='sims'):
        """[summary]

        Parameters
        ----------
        func : function
            main function to simulate
            cannot contain *args or **kwargs
        sim_folder : str
            the path to store the result, by default 'sims'            
        """
        self.func=func
        self.len_t_vec=len_t_vec
        self.sim_folder=sim_folder
        self.stored_title=os.path.split(sim_folder)[-1]
        if sim_folder!=None:
            if not os.path.exists(sim_folder):
                os.mkdir(sim_folder)
        parameters=inspect.signature(func).parameters
        
        self.parameters_name=list(parameters.keys())
        self.parameters=dict.fromkeys(self.parameters_name,None)
        self.iter_parameters=dict()
        self.need_to_set=set() # .add/ .discard
        self.need_to_iter=set()
        self.already_set=set()

        for key in self.parameters_name:
            if self.parameters[key]==inspect._empty:
                self.need_to_set.add(key)
            else:
                self.parameters[key]=parameters[key].default
                self.already_set.add(key)
    def show_need_to_set(self):
        print(self.need_to_set)
    def show_already_set(self):
        print(self.already_set)
    def show_iter_conditions(self):
        print(list(self.iter_parameters.keys()))
        print(list(itertools.product(*self.iter_parameters.values())))


    def set_fix_param(self,**kwargs):
        for key,vals in kwargs.items():
            if key=='a' and self.len_t_vec>0:
                self.parameters['a']=vals*np.ones(self.len_t_vec)
                self.parameters['ana']=vals*np.ones(self.len_t_vec)
            else:
                self.parameters[key]=vals
            self.already_set.add(key)
            self.need_to_set.discard(key)

    def set_iter_param(self,**kwargs):
        for key,vals in kwargs.items():
            self.iter_parameters[key]=vals
            self.already_set.add(key)
            self.need_to_set.discard(key)

    def run(self,start=0,cross=False,save=True):
        
        func= itertools.product if cross else zip
        for i,vals in enumerate(tqdm(func(*self.iter_parameters.values()))):
            print(i,list(self.iter_parameters.keys()),vals)
            if i>=start:
                for j,key in enumerate(self.iter_parameters.keys()):
                    if key=='a' and self.len_t_vec>0:
                        self.parameters['a']=vals[j]*np.ones(self.len_t_vec)
                        self.parameters['ana']=vals[j]*np.ones(self.len_t_vec)
                    else:
                        self.parameters[key]=vals[j]
                    
                res={'param':self.parameters,'data':self.func(**self.parameters)}
                if save:
                    np.save(os.path.join(self.sim_folder,f'{self.stored_title}_{i}.npy'),res)

class RSimulation:
    """
    read simulation
    """
    def __init__(self,func,sim_folder):
        """[summary]

        Parameters
        ----------
        func : function
            main function to simulate
            cannot contain *args or **kwargs
        sim_folder : str
            the path to store the result, by default 'sims'  
        """
        filenames=sorted(glob.glob(f'{sim_folder}/*.npy'))
        
        self.returned_names=func.__code__.co_varnames[func.__code__.co_argcount:]
        self.params=[]
        self.datas=[]
        for f in filenames:
            tmp=np.load(f,allow_pickle=True).item()
            self.params.append(tmp['param'])
            self.datas.append(tmp['data'])










