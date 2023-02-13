'''
Filename: /home/flumer/ION/MySoft/SimuFit/src/utils.py
Path: /home/flumer/ION/MySoft/SimuFit/src
Created Date: Sunday, July 19th 2020, 1:57:35 pm
Author: flumer

'''

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
aa=np.asarray



def plot_dist(t_vec,g_upper,g_lower,g_inner_upper=None,g_inner_lower=None,g_inner=None,normalize=False,save=None):

    if normalize:
        g_upper_n=g_upper.sum()
        g_lower_n=g_lower.sum()
        g_inner_upper_n=g_inner_upper.sum() if g_inner_lower is not None else None
        g_inner_lower_n=g_inner_lower.sum() if g_inner_lower is not None else None
        g_inner_n=g_inner.sum() if g_inner is not None else None
            
    else:
        g_upper_n=1
        g_lower_n=1
        g_inner_n=1
        g_inner_upper_n=1
        g_inner_lower_n=1
    

    fig,axes=plt.subplots(nrows=2,ncols=2,sharex=True)
    axes[0,0].plot(t_vec,g_upper/g_upper_n,label='upper')
    if g_inner_upper is not None:
        axes[0,0].plot(t_vec,g_inner_upper/g_inner_upper_n,label='inner_upper')
    axes[0,0].legend()
    axes[0,1].plot(t_vec,g_lower/g_lower_n,label='lower')
    if g_inner_lower is not None:
        axes[0,1].plot(t_vec,g_inner_lower/g_inner_lower_n,label='inner_lower')
    axes[0,1].legend()
    
    axes[1,0].plot(t_vec,g_upper/g_upper_n,label='upper')
    axes[1,0].plot(t_vec,g_lower/g_lower_n,label='lower')
    if g_inner is not None:
        axes[1,0].plot(t_vec,g_inner/g_inner_n,label='inner')
    axes[1,0].legend()
    if g_inner_upper is not None and g_inner_lower is not None:
        axes[1,1].plot(t_vec,g_inner_upper/g_inner_upper_n,label='inner_upper')
        axes[1,1].plot(t_vec,g_inner_lower/g_inner_lower_n,label='inner_lower')
        axes[1,1].legend()
    if save!=None:
        plt.savefig(save)
    else:
        plt.show()

@njit
def runmed(unfiltered,filter_width,first_value,nan_handling):
    """running median filter

    Parameters
    ----------
    unfiltered : NDArray[float]
        unfiltered data row or column vector length >= filter_width
    filter_width : float
        width of the median filter towards the vector border the width is
    reduced
    first_value : int
        first value to be calculated the remaining boundary values are calculated with NaNs 
        filled up
    nan_handling : int
        0: if there is a np.nan within the filter window, the filtered
           value also np.nan
        1: NaNs in the filter window are ignored and the calculation with the
          remaining values executed
    Returns
    -------
    NDArray[float]
        filtered data (row vector)
    """
   

    n=len(unfiltered)    

    if filter_width<1:
        raise Exception('filter_width too small!')
    
    if np.floor(filter_width/2)*2==filter_width:
        raise Exception('filter_width must be odd!')
    
    if n<filter_width:
        raise Exception('Vector too short!')
    

    if first_value>np.floor((n+1)/2):
        raise Exception('First value selected too large!')
    
    if first_value<1:
        raise Exception('First value selected too small!')
    
    if round(first_value)!=first_value:
        raise Exception('First value must be an integer!')

    if first_value>(filter_width+1)/2:
        raise Exception('First value selected too large!')
    

    last_value=n-first_value+1

    if not nan_handling in [0,1]:
        raise Exception('Invalid value for np.nan handling!')
    

    filtered=np.zeros(n)
    # Margins
    if first_value>1:
        filtered[:first_value]=np.nan
        filtered[last_value+1:]=np.nan

    start_full=int((filter_width+1)/2)
    half_width=int((filter_width-1)/2)

    if nan_handling==0: # no np.nan correction

        # Range with full filter_width
        for i in range(start_full-1,n-start_full+1):
            filtered[i]=np.median(unfiltered[i-half_width:i+half_width+1])
    
    
        # intermediate ranges
        if first_value<start_full:
            for i in range(first_value-1,start_full-1):
                filtered[i]=np.median(unfiltered[:2*i+1])
            for i in range(n-start_full+1,last_value):
                filtered[i]=np.median(unfiltered[2*i+1-n:])
    else: # np.nan correction
        # Range with full filter_width
        for i in range(start_full-1,n-start_full+1):
            vector=unfiltered[i-half_width:i+half_width+1]
            useful_data=vector[~np.isnan(vector)]
            if len(useful_data)==0:
                filtered[i]=np.nan
            else:
                filtered[i]=np.median(useful_data)    
        # intermediate ranges
        if first_value<start_full:
            for i in range(first_value-1,start_full-1):
                vector=unfiltered[:2*i+1]
                useful_data=vector[~np.isnan(vector)]
                if len(useful_data)==0:
                    filtered[i]=np.nan
                else:
                    filtered[i]=np.median(useful_data)
                
            for i in range(n-start_full+1,last_value):
                vector=unfiltered[2*i+1-n:]
                useful_data=vector[~np.isnan(vector)]
                if len(useful_data)==0:
                    filtered[i]=np.nan
                else:
                    filtered[i]=np.median(useful_data)
    return filtered

# for graph and statistics


# def KLdivergence(p,q):
#     return -np.sum(np.nan_to_num(p*np.log(q/p)))

# #%% for plotting
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# splot=lambda:'./neuron-plot.mplstyle'
# smat=lambda:'./neuron-mat.mplstyle'
# smultiplot=lambda:'./neuron-multiplot.mplstyle'


# class SF(object):
#     """
#     save figure context
#     """
#     def __init__(self,figure_name,figures_folder='model_figures',style=splot(),**fig_kwargs):
#         plt.style.use(style)
#         self.figure_folder=figures_folder
#         self.figure_name=figure_name
#         self.figure_type='svg'
#         self.fig_kwargs=fig_kwargs
#         if not os.path.exists(self.figure_folder):
#             os.mkdir(self.figure_folder)
#     def __enter__(self):
#         self.fig,self.axes=plt.subplots(**self.fig_kwargs)
#         return self.fig,self.axes
#     def __exit__(self,exc_type, exc_value, exc_traceback):
#         # plt.tight_layout()
#         if not (self.figure_name is None):
#             self.fig.suptitle('') # by default, remove suptitle in figure
#             if not isinstance(self.axes,np.ndarray):
#                 self.axes.set_title('') # by default, remove title in axes
#             self.fig.savefig('%s/%s.%s'%(self.figure_folder,self.figure_name,self.figure_type),format=self.figure_type,bbox_inches='tight')