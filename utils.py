
'''
Filename: /home/flumer/Documents/Project/to_lxd_7subj_20201025/utils.py
Path: /home/flumer/Documents/Project/to_lxd_7subj_20201025
Created Date: Monday, October 26th 2020, 9:49:57 am
Author: LI Xiaodong

Copyright (c) 2020 Your Company
'''

import glob,os
from numpy.core.multiarray import result_type
import pandas as pd
import numpy as np
from scipy import stats

import re,itertools

from scipy.stats.stats import mode

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from dateutil.parser import parse
import settings

aa=np.asarray
elinewidth=0.5
capsize=1.5

def label_diff(ax,text,xpos,ypos,shrink_x=0.,base_y=0.,increment_y=0.,fontsize=12):
    """[summary]

    Parameters
    ----------
    ax : matplotlib.axes
        [description]
    text : str
        text to show
    xpos : float/list/tuple/np.ndarray[float]
        if len(x)=2 then it is supposed to be diff and x[0]<x[1]
        else it is supposed to be single tick

    ypos : float
        y position
    shrink_x : float, optional
        shrink x position, by default 0.
    base_y : float, optional
        y base, by default 0.
    increment_y : float, optional
        y increment, by default 0.
    fontsize : float, optional
        the fontsize of two difference
    """    
    delta=base_y+increment_y
    if not isinstance(xpos,float) and not isinstance(xpos,int):
        x0,x1=xpos
        x0+=shrink_x;x1-=shrink_x
        xc=(x0+x1)/2
        ax.plot([x0,x0,x1,x1],[ypos+base_y,ypos+delta,ypos+delta,ypos+base_y],'k',linewidth=1)
    else:
        xc=xpos
    ax.text(xc,ypos+delta,text,ha='center',va='baseline',color='k',fontsize=fontsize)

        
def axSet(ax,title=None,
          xticks=None,yticks=None,xticklabels=None,yticklabels=None,
          xlabel=None,ylabel=None,xlim=None,ylim=None,legend=True,grid=False,
          extra_options=dict()):
    """
        setting parameters related to axis
    """
    if grid:
        ax.grid()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if legend is not None and legend is not False:
        if 'legend' in extra_options.keys():
            ax.legend(numpoints=1,bbox_to_anchor=(1,1),**extra_options['legend'])
        else:
            ax.legend(numpoints=1,bbox_to_anchor=(1,1))

def figSet(fig,ax=None,title=None,xticks=None,yticks=None,
          xlabel=None,ylabel=None,xlim=None,ylim=None,legend=None,extra_options=dict()):
    if ax is not None:
        
        handles, labels = ax.get_legend_handles_labels()
        labels=labels if legend is None else legend
        if 'legend' in extra_options.keys():
            fig.legend(handles, labels,**extra_options['legend'])
        else:
            fig.legend(handles, labels)

        suptitle=ax.get_title() if title is None else title
        fig.suptitle(suptitle)

        if xlabel is None:
            ...
        else:
            fig.supxlabel(xlabel)
            
        if ylabel is None:
            ...
        else:
            fig.supylabel(ylabel)

class BaseData:
    
    def __init__(self,folder,subject_name,coherences=[0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26],data_type=None):
        """
        read rearranged data which suits for analysis
        
        Parameters
        ----------
        folder : str
            a folder which stores all rearanged subjects files which suits for analysis.
        subject_name : str
            a specific subject
        """
        if data_type is None:raise Exception("`data_type` not specified")
        
        self.folder=folder
        self.cohs=aa(coherences)
        self.dircohs=np.r_[-np.flip(self.cohs[1:]),self.cohs]
        
        self.subject=subject_name.split('_')[0]
        self.old_remove_up026=False
        self._df_stored=dict()

        self.df=self._extraCols(pd.read_csv(
            "%s/%s/%s_%s.csv"%(folder,data_type,subject_name,data_type)))
        

    def _extraCols(self,df):
        """create extra columns for the DataFrames
        """
        
        self.added_cols=['_chDir','_up','_correct','_wrong','wrong','_lenme','firstStageRtGr']
        def to_num(x,term0,term1):
            if x==term0:
                return 0
            elif x==term1:
                return 1
            else:
                return 0.5
            
        df.loc[:,'_chDir']=df['choosedDirection'].apply(
            lambda x:to_num(x,'left','right')).astype(float)

        df.loc[:,'_up']=df['intoStage2'].apply(
            lambda x:to_num(x,False,True)).astype(float)

        df.loc[:,'_correct']=df[['intoStage2','correct']].apply(
            lambda x:1 if x['correct']==True and x['intoStage2']==False else 0,axis=1).astype(float)
        
        df.loc[:,'_wrong']=df[['intoStage2','correct']].apply(
            lambda x:1 if x['correct']==False and x['intoStage2']==False else 0,axis=1).astype(float)
        df.loc[:,'wrong']=df['correct'].apply(lambda x:~x if isinstance(x,bool) else x)
        
        df.loc[:,'_firstStageRtGr5']=pd.qcut(df['firstStageRt'].to_numpy(),q=5).codes

        self._up_RTmean={
            'dots.coherence':df.groupby('dots.coherence')['firstStageRt'].mean(),
            'dots.dircoherence':df.groupby('dots.dircoherence')['firstStageRt'].mean()}

        return df

    def extractData(self,condition,remove_up026=False,remove_quick=0.25):
        """extract sub dataframe conditioned on `condition`

        Parameters
        ----------
        condition : str
            a string spilt by one space,
            the first term of the `condition` defines trial type
            the seocnd term of the `condition` define stage
        remove_up026 : bool
            remove opt-up 0.26
        remove_quick ： float
            remove quick reaction time trials lower than `remove_quick`
        Returns
        -------
        pd.DataFrame
            extracted dataframe

        Raises
        ------
        Exception
            when the first term is not one of ['no-up','opt-up','nodelay','delay*']
        Exception
            when the second term is not one of ['lr','up','2nd','quick','lr-right','lr-wrong']
        """
        conds=condition.split(' ')
        assert(len(conds)<=3)
        if len(conds)==2:
            trial_type,stage=conds
            correctnot=None
        elif len(conds)==3:
            trial_type,stage,correctnot=conds
        else:
            trial_type=conds[0]
            stage=None
            correctnot=None

        if condition in self._df_stored:
            if stage=='up':
                if self.old_remove_up026==remove_up026:
                    return self._df_stored[condition],trial_type,stage
            else:
                self.old_remove_up026=remove_up026
                
                    
        query_statement=[]

        if trial_type in ['no-up','opt-up']:
            _trialType=[trial_type]
            _delayTime=0
        elif trial_type=='nodelay':
            _trialType=['no-up','opt-up']
            _delayTime=0
        elif trial_type=='delay':
            _trialType=['opt-up']
            _delayTime=[0.25,0.5,1.]
        elif trial_type[:5]=='delay':
            _trialType=['opt-up']
            _delayTime=int(trial_type[5:])/1000
        else:
            raise Exception("illegal first term")
        query_statement.extend([
            "trialType==@_trialType",
            "delayTime==@_delayTime",
            "firstStageRt>=@remove_quick"
            ])
        
        if stage is not None:
            if stage in ['lr','quick','slow']:
                _intoStage2=False
            elif stage in ['up','2nd']:
                _intoStage2=True

                if stage=='up' and remove_up026:
                    query_statement.append("`dots.coherence`!=0.26")
            else:
                raise Exception("illegal second term")
            query_statement.append("intoStage2==%r"%_intoStage2)

        if correctnot is not None:
            if correctnot=='correct':
                _correct=True    
            elif correctnot=='error':
                _correct=False
            else:
                raise Exception("illegal third term")
            query_statement.append("correct==%r"%_correct)

        ret_df=self.df.query(' & '.join(query_statement))
    
        self._df_stored[condition]=ret_df
        return ret_df,trial_type,stage

class AnalysisData(BaseData):

    data_type='analysis'
    def __init__(self,folder,subject_name,coherences=[
        0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26]):
        super(AnalysisData,self).__init__(folder,subject_name,coherences,self.data_type)


    def plot_proportion(self,ax,data=['opt-up'],prop_types=['correct','wrong','up'],dirc=True,under='coh',label_prop=False,errorbar=False,joinline=False,comparable=False,avg=False,remove_up026=False,alpha=0.5):
        """plot proportion

        Parameters
        ----------
        ax : matplotlib axes
            [description]
        data : list[str], optional
            different condition of data, by default ['opt-up']
        prop_types : list of str, optional
            proportion type ['correct','wrong','up'], by default ['correct']
        dirc : bool, optional
            direction/absolute coherence or not, by default True
        under : str, optional
            under choosed result or coherences, by default 'coh'
        label_prop : bool, optional
            legend contain prop_type        
        errorbar : bool, optional
            [description], by default False
        joinline : bool, optional
            join lines in error bar, by default False
        comparable : bool, optional
            correct rate is computed by correct/(correct+wrong+up) if False
            else correct/(correct+wrong)
            , by default False
        avg : bool optional
            avg mode or not
        """        
        for d in data:

            gr='dots.dircoherence' if dirc else 'dots.coherence'
            
            for pt in prop_types:
                
                if pt=='up' and avg: # average mode
                    g,trial_type,_=self.extractData("%s lr"%d,remove_up026=remove_up026)
                else:
                    g,trial_type,_=self.extractData(d,remove_up026=remove_up026)

                if g.empty:continue
                
                if trial_type=='no-up' and pt=='up': continue
                
                if pt=='correct': 
                    se='_correct' if not comparable else 'correct'
                    if d=='opt-up up':d="opt-up 2nd" # change the tag
                elif pt=='up': se='_up'
                elif pt=='wrong': se='_wrong' if not comparable else 'wrong'
                
                if under=='coh':
                    prop=g.groupby(gr)[se].apply(lambda x:x.sum()/len(x))
                elif under=='result':
                    prop=g.groupby(gr)[se].sum()
                    prop/=prop.sum()
            
                if label_prop:
                    label=f"{settings.label_map[d]} {settings.label_map[pt]}"
                else:
                    label=f"{settings.label_map[d]}"
                
                fmt='.-' if joinline else 'o'
                if errorbar:
                    prop_sem=g.groupby(gr)[se].sem()
                    
                    ax.errorbar(prop.index,prop,prop_sem,fmt=fmt,color=settings.color_map[d],label=label,alpha=alpha,elinewidth=elinewidth,capsize=capsize)
                else:
                    ax.plot(prop.index,prop,fmt,color=settings.color_map[d],label=label,alpha=alpha)

    def plot_responseTime(self,ax,data=['opt-up lr','opt-up up','no-up lr'],method='errorbar',dirc=True,cohs=None,joinline=False,with_scatter=False,with_hist=False,remove_up026=False):
        """plot response time

        Parameters
        ----------
        ax : matplotlib axes
            [description]
        data : list, optional
            different condition of data, by default ['opt-up lr','opt-up up','no-up lr']
        method : str, optional
            'errorbar', for each data, plotting its response time using errorbar
            'hist-raw', for each data, plotting its response time using hist
            'hist-fine', plotting every coherence response time
            by default 'errorbar'
        dirc : bool, optional
            direction/absolute coherence or not , by default True
        cohs : list, optional
            specify coherence,only work when method='hist-fine', by default None, if all coherences are needed.
        joinline : bool, optional
            join plot lines, by default False.
        with_scatter bool, optional
            with scatter or not, only works when method='errorbar' , by default False
        with_scatter bool, optional
            with hist or not, only works when method='hist-*' , by default False
        """        
        for d in data:
            g,_,stage=self.extractData(d,remove_up026)
            if g.empty:continue

            rt_stage='secondStageRt' if stage=='2nd' else 'firstStageRt'
            if np.isnan(g[rt_stage]).all():continue
            
            if stage=='2nd':   
                # gr='dots.dirfinalcoherence' if dirc else 'dots.finalcoherence' # second stage has only one coherence for now
                gr='dots.dircoherence' if dirc else 'dots.coherence'                
            else:
                gr='dots.dircoherence' if dirc else 'dots.coherence'
            
            if stage=='quick':
                valid_trials=[]
                for i,iv in enumerate(self._up_RTmean[gr].iteritems()):
                    index,value=iv
                    if i==0:
                        valid_trials=(g[gr]==index) & (g[rt_stage]<value)
                    else:
                        valid_trials=valid_trials | ((g[gr]==index) & (g[rt_stage]<value))
                g=g[valid_trials]
            
            if stage=='slow':
                rt_mean=g.groupby(gr)[rt_stage].mean()
                valid_trials=[]
                for i,iv in enumerate(self._up_RTmean[gr].iteritems()):
                    index,value=iv
                    if i==0:
                        valid_trials=(g[gr]==index) & (g[rt_stage]>value)
                    else:
                        valid_trials=valid_trials | ((g[gr]==index) & (g[rt_stage]>value))
                g=g[valid_trials]

            if method=='errorbar': 
                rt=g.groupby(gr)[rt_stage].agg(["mean","sem"])
                fmt='.-' if joinline else 'o'
                ax.errorbar(rt.index,rt['mean'],rt['sem'],fmt=fmt,color=settings.color_map[d],label=settings.label_map[d],alpha=0.5,elinewidth=elinewidth,capsize=capsize)
                # ax.fill_between(rt.index,rt['mean']-rt['sem'],rt['mean']+rt['sem'],alpha=0.5)
                if with_scatter:
                    ax.scatter(gr,rt_stage,data=g,color=settings.color_map[d],alpha=0.35,label=None)
            elif method=='hist-raw':
                sns.distplot(g[rt_stage],hist=with_hist,color=settings.color_map[d],
                label=settings.label_map[d],
                hist_kws=dict(edgecolor="black", linewidth=1, alpha=0.8),
                ax=ax)
                # ax.get_legend().remove() # sns.distplot contains legend by default
                ax.set_xlabel('')
            elif method=='hist-fine':
                for coh,g_coh in g.groupby(gr):
                    if cohs is not None and not coh in cohs:continue 
                    sns.distplot(g_coh[rt_stage],hist=with_hist,color=settings.get_color(coh,d),label="%s-%.2f"%(settings.label_map[d],coh),ax=ax)
                    ax.get_legend().remove() # sns.distplot contains legend by default
                    ax.set_xlabel('')
    
    def plot_rtAcc(self,ax,data=['no-up lr','opt-up'],bins=5,dirc=False,coh=None,stat=False,joinline=True,remove_up026=False,mc=False):
        
        for_ttests=[]
        for d in data:
            g,_,stage=self.extractData(d,remove_up026)
            if g.empty:continue
            dots_coh='dots.dircoherence' if dirc else 'dots.coherence'
            
            if coh==None:
                g_coh=g.copy()
            else:
                g_coh=g[g[dots_coh]==coh]
                
            gr=pd.qcut(g_coh["firstStageRt"], q=bins)
            corr_gr=[group.to_numpy() for _,group in g_coh.groupby(gr)['correct']]

            Correct=g_coh.groupby(gr)['correct'].mean()
            SEM_Correct=g_coh.groupby(gr)['correct'].sem()

            fmt='-o' if joinline else 'o'
            if coh is None:
                ax.errorbar(np.arange(bins),Correct,SEM_Correct,fmt=fmt,color=settings.color_map[d],label="%s"%settings.label_map[d],alpha=0.5,elinewidth=elinewidth,capsize=capsize)
            else:
                ax.errorbar(np.arange(bins),Correct,SEM_Correct,fmt=fmt,color=settings.get_color(coh,d),label="%s-%.2f"%(settings.label_map[d],coh),alpha=0.5,elinewidth=elinewidth,capsize=capsize)
            for_ttests.append(corr_gr)

        if stat:
            ps=[];sigs=[]
            for i in range(bins):
                _,p=stats.mannwhitneyu(for_ttests[0][i],for_ttests[1][i],alternative='less')
                ypos=np.maximum(np.max(for_ttests[0][i]),np.max(for_ttests[1][i]))
                ps.append(p) # print out the p value
                sigs.append(p<0.05)                
            if mc:
                sigs=multipletests(ps,alpha=0.05,method='bonferroni')[0]
            for i,sig in enumerate(sigs):
                l='*' if sig else '-'
                label_diff(ax,l,i,ypos)
            
            # print(ps)

    def plot_rtCohFit(self,ax,data=['opt-up up'],box=False,quantile=False,show_p=True,remove_up026=False):
        """plot response time and coherences fit

        Parameters
        ----------
        ax : matplotlib axes
            [description]
        data : list, optional
            different condition of data, by default ['opt-up up']
        box : bool, optional
            scatter or box plot
            box representation or not, by default False
        quantile : bool, optional
            show quantile regression plot or not, by default False
        show_p : bool, optional
            show pvalue
        Returns
        -------
        [type]
            [description]
        """
        x_scale=100 if box else 1
        for d in data:
            g,_,stage=self.extractData(d,remove_up026)
            if g.empty:continue
            rt_stage='secondStageRt' if stage=='2nd' else 'firstStageRt'
            
            Qdot_coh="Q('dots.coherence')"

            if quantile:
                quantiles = np.arange(.2, 1.1, .2)
                mod = smf.quantreg(f"{rt_stage} ~ {Qdot_coh}", data=g)
                def fit_model(q):
                    res = mod.fit(q=q,disp=0)
                    return [q, res.params['Intercept'], res.params[Qdot_coh]] + res.conf_int().loc[Qdot_coh].tolist()
                models = [fit_model(x) for x in quantiles]
                models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

            ols = smf.ols(f"{rt_stage} ~ {Qdot_coh}", data=g).fit()
            slope=ols.params[Qdot_coh]
            if ols.pvalues[Qdot_coh]<0.01:
                star='***'
            elif ols.pvalues[Qdot_coh]<0.05:
                star='**'
            elif ols.pvalues[Qdot_coh]<0.1:
                star='*'
            else:
                star='-'
                
            ols_ci = ols.conf_int().loc[Qdot_coh].tolist()
            ols = dict(a = ols.params['Intercept'],
                    b = ols.params[Qdot_coh],
                    lb = ols_ci[0],
                    ub = ols_ci[1],
                    pvalue=ols.pvalues[Qdot_coh])

            gcoh=g['dots.coherence']
            x = np.linspace(gcoh.min(), gcoh.max(), 50)
            get_y = lambda a, b: a + b * x

            if quantile:
                for i in range(models.shape[0]):
                    y = get_y(models.a[i], models.b[i])
                    ax.plot(x*x_scale, y, '-.', color=settings.color_map[d])
    

            y = get_y(ols['a'], ols['b'])
            

            ax.plot(x*x_scale, y, color=settings.color_map[d], label='%s OLS slope=%.2f(%s)'%(settings.label_map[d],slope,star))
            
            if box:
                df_box=pd.DataFrame(aa([gcoh,g[rt_stage]]).T,columns=['coh','rt'])
                cohs=[]
                cohs_rt=[]
                for coh,gbox in df_box.groupby('coh'):
                    cohs.append(coh)
                    cohs_rt.append(gbox['rt'])
            
                ax.boxplot(cohs_rt,positions=aa(cohs)*x_scale,showfliers=False,widths=0.5)
                ax.set_xticks(aa(cohs)*x_scale)
                ax.set_xticklabels([str(coh)[1:] for coh in cohs])
            else:
                ax.scatter(gcoh, g[rt_stage], alpha=.2,color='grey')
            
            if show_p:
                ax.text(0.9, 1,'p=%.2f'%ols['pvalue'], ha='center', va='center', transform=ax.transAxes)
    
    def plot_rtCohFitResult(self,ax,data=['opt-up up'],param=False,remove_up026=False):
        for d in data:
            g,_,stage=self.extractData(d,remove_up026)
            if g.empty:continue

            rt_stage='secondStageRt' if stage=='2nd' else 'firstStageRt'
            
            Qdot_coh="Q('dots.coherence')"
            y=[]
            for coh in self.cohs:
                ng=g[g['dots.coherence']<=coh]
            
                ols = smf.ols("%s ~ %s"%(rt_stage,Qdot_coh), data=ng).fit()
                y.append(ols.params[Qdot_coh] if param else ols.pvalues[Qdot_coh])

            
            if not param:
                y=np.log10(y)
                ax.plot(self.cohs,-1*np.ones_like(self.cohs),'--',color='black')

            ax.plot(self.cohs, y, '.-',color=settings.color_map[d], label=settings.label_map[d])
            ax.plot(self.cohs,np.zeros_like(y),'--',color='k')
    
    def plot_psychometricCurve(self,ax,data=['no-up lr','opt-up lr'],errorbar=False,with_scatter=False):
        """plot psychometric curve

        Parameters
        ----------
        ax : matplotlib axes
            [description]
        data : list, optional
            different condition of data, by default ['no-up lr','opt-up lr']
        with_scatter : bool, optional
            with scatter or not, by default True
        """        
        for d in data:
            g,_,_=self.extractData(d)
            if g.empty:continue
            
            y=g.loc[:,'_chDir']
            x=sm.add_constant(g['dots.dircoherence'])
        
            model=sm.Logit(y,x).fit(disp=False)
            right_logit=model.predict(sm.add_constant(self.dircohs))
            slope=model.params['dots.dircoherence']
            
            right_prop=g.groupby('dots.dircoherence')['_chDir'].apply(lambda x:x.sum()/len(x))
            
            if errorbar:
                right_prop_sem=g.groupby('dots.dircoherence')['_chDir'].sem()
                ax.errorbar(self.dircohs,right_logit,right_prop_sem,fmt='.-',color=settings.color_map[d],label='%s slope=%.2f'%(settings.label_map[d],slope),alpha=0.6,elinewidth=elinewidth,capsize=capsize)
            else:
                ax.plot(self.dircohs,right_logit,'.-',color=settings.color_map[d],label='%s slope=%.2f'%(settings.label_map[d],slope),alpha=0.6)            

            if with_scatter:
                ax.scatter(right_prop.index,right_prop,color=settings.color_map[d],alpha=0.35)
    
    def stat_pairData(self,dirc=True,data=['opt-up lr','opt-up up'],term='firstStageRt',cohs=None):
        
        assert(len(data)<=2)
        
        block=[]
        if dirc:
            cohs=self.dircohs if cohs is None else cohs.copy()
            dot_coh='dots.dircoherence'
        else:
            cohs=self.cohs if cohs is None else cohs.copy()
            dot_coh='dots.coherence'
        for d in data:
            g,_,stage=self.extractData(d)
            tm='secondStageRt' if stage=='2nd' else term 
            block.append(g[g[dot_coh].isin(cohs)][[dot_coh,tm]].rename({tm:"%s_%s"%(d,tm)}))
        
        return block

    def plot_rtVar(self,ax,data=['no-up lr','opt-up lr','opt-up up'],alpha=1,remove_up026=False):

        var_mat=np.zeros(len(data))

        for j,d in enumerate(data):
            g,_,_=self.extractData(d,remove_up026=remove_up026)
            if g.empty:continue
            var_mat[j]=g['firstStageRt'].var()

        idx=np.arange(len(data)).astype(int)
        ax.plot(idx,var_mat.T,'.-',color='k',alpha=alpha)
        ax.set_xticks(idx)
        ax.set_xticklabels([settings.label_map[d] for d in data])


class AnalysisAvgData(AnalysisData):
    data_type='analysis'

    def __init__(self,folder,subject_names,coherences=[0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26],
        averaged_by='coh',remove_quick=0.25):
        """read rearranged data which suits for analysis
        
        Parameters
        ----------
        folder : str
            a folder which stores all rearanged subjects files which suits for analysis.
        subject_name : str
            a specific subject
        coherences : list[float]
            coherence levels
        coherences : list, optional
            [description], by default [0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26]
        averaged_by : str, optional
            for each subject the data what is averaged by ['coh','nodircoh','RT'], by default 'coh'
        remove_quick : float, optional
            remove quick response data, by default 0.25
        """        
        self.averaged_by=averaged_by
        self.df=[]
        for subject_name in subject_names:
            tmpdf=pd.read_csv(
                "%s/%s/%s_%s.csv"%(folder,self.data_type,subject_name,self.data_type))
            tmpdf=tmpdf.query("firstStageRt>=@remove_quick")
            df=self._avgSubject(tmpdf,subject_name)
            self.df.append(df)
        self.df=pd.concat(self.df,ignore_index=True)
        
        self.cohs=aa(coherences)
        self.subjects=subject_names
        self.dircohs=np.r_[-np.flip(self.cohs[1:]),self.cohs]
        self._df_stored=dict()
    
    def _avgSubject(self,df,subject):

        tmpdf=self._extraCols(df)

        if self.averaged_by=='coh':
            avgCols=[['_chDir','_correct','_wrong','correct','wrong','firstStageRt','secondStageRt'],
                    ['_up']]
            dfs=[pd.DataFrame(),pd.DataFrame()]
            gr=[['trialType','dots.dircoherence','intoStage2','delayTime'],
                ['trialType','dots.dircoherence','delayTime']]
        elif self.averaged_by=='nodircoh':
            avgCols=[['_chDir','_correct','_wrong','correct','wrong','firstStageRt','secondStageRt'],
                ['_up']]
            dfs=[pd.DataFrame(),pd.DataFrame()]
            gr=[['trialType','dots.coherence','intoStage2','delayTime'],
                ['trialType','dots.coherence','delayTime']]
        elif self.averaged_by=='RT':
            avgCols=[['correct','firstStageRt','secondStageRt']]
            dfs=[pd.DataFrame()]
            gr=[['trialType','intoStage2','_firstStageRtGr5','delayTime']]
        
        # raw processing
        for i,df in enumerate(dfs):
            for a in avgCols[i]:
                df[a]=tmpdf.groupby(gr[i],dropna=False)[a].mean()
            df['trial_count']=tmpdf.groupby(gr[i],dropna=False).size()
            df.index=pd.MultiIndex.from_tuples(df.index)
            dfs[i]=df.rename_axis(index=gr[i]).reset_index()
        
        # merge all subjects
        df_m=dfs[0]
        for i,df in enumerate(dfs[1:]):
            on=list(set(gr[0]) & set(gr[i+1]))
            df_m=df_m.merge(df,left_on=on,right_on=on,suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        # regenerate necessary columns
        df_m['subject']=subject
        if 'dots.dircoherence' in df_m.columns:
            df_m['dots.coherence']=abs(df_m['dots.dircoherence'])
        
        return df_m
    
    def extractData(self,condition,remove_up026=False):
        conds=condition.split(' ')
        assert(len(conds)<3)
        if len(conds)>1:
            if not conds[1] in ['lr','up','2nd']:
                raise Exception('illegal second term')
        df=super(AnalysisAvgData, self).extractData(condition,remove_up026)
        
        return df

    def plot_rtAcc(self,ax,data=['no-up lr','opt-up'],stat=False,remove_up026=False,fontsize=12,fmt='.-',mc=False):
        for_ttests=[]
        bins=5
        maxy=-np.inf
        for d in data:
            g,_,stage=self.extractData(d,remove_up026)
            if g.empty:continue
                
            grp=g.groupby('_firstStageRtGr5')['correct']
            corr_gr=[group.to_numpy() for _,group in grp]

            Correct=g.groupby('_firstStageRtGr5')['correct'].mean()
            SEM_Correct=g.groupby('_firstStageRtGr5')['correct'].sem()
            maxy=np.max(np.maximum(Correct+SEM_Correct,maxy))

            ax.errorbar(np.arange(bins),Correct,SEM_Correct,fmt=fmt,color=settings.color_map[d],label="%s"%settings.label_map[d],alpha=0.5,elinewidth=elinewidth,capsize=capsize)
            for_ttests.append(corr_gr)
            ax.set_xticks(np.arange(bins))
            ax.set_xticklabels(np.arange(bins)+1)

        if stat:
            ps=[];sigs=[]
            for i in range(bins):
                _,p=stats.mannwhitneyu(for_ttests[0][i],for_ttests[1][i],alternative='less')
                ps.append(p)
                sigs.append(p<0.05)
            if mc:
                sigs=multipletests(ps,alpha=0.05,method='bonferroni')[0]
            for i,sig in enumerate(sigs):
                l='*' if sig else '-'
                label_diff(ax,l,i,maxy,fontsize=fontsize)
            
            # print(ps)
class AnalysisIterData(AnalysisData):
    data_type='analysis'    
    def __init__(self,folder,subject_names,coherences=[
        0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26]):
        """
        read rearranged data which suits for analysis
        
        Parameters
        ----------
        folder : str
            a folder which stores all rearanged subjects files which suits for analysis.
        subject_name : str
            a specific subject
        """
        self.folder=folder
        self.subject_names=subject_names
        self.coherences=coherences

    def show_trialsNumber(self,data=['no-up','opt-up','opt-up lr','opt-up up'],remove_up026=False):
        """show number of trials for each condition

        Parameters
        ----------
        data : list, optional
            [description], by default ['no-up','opt-up','opt-up lr','opt-up up']
        remove_up026 : bool, optional
            [description], by default False

        Returns
        -------
        [type]
            [description]
        """
        trial_numbers=[]
        
        for i,subject_name in enumerate(self.subject_names):
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)
            trial_number=[]
            for d in data:
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                trial_number.append(g.shape[0])
            trial_number.append(sum(trial_number[:2])) #total
            trial_numbers.append(trial_number)
        
        return pd.DataFrame(trial_numbers,columns=data+['total'],index=np.arange(1,len(self.subject_names)+1))

    def plot_rtAccSlope(self,ax,data=['no-up lr','opt-up lr'],RT_type=None,remove_up026=False,stat=False):
        """[summary]

        Parameters
        ----------
        ax : [type]
            [description]
        data : list, optional
            [description], by default ['no-up lr','opt-up lr']
        RT_type : str, optional
            'group':using grouped RT
            'standardize': using standardized RT
            , by default None
        remove_up026 : bool, optional
            [description], by default False
        stat : bool, optional
            [description], by default False
        """        
        
        slope_noup=[]
        slope_opt=[]
        
        for subject_name in self.subject_names:
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for d in data:
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                g['correct']=g['correct'].astype(int)
                if RT_type=='group':
                    term='_firstStageRtGr5'
                    reg=g.groupby(term)['correct'].mean().reset_index()
                elif RT_type=='standardize':
                    term='firstStageRt'
                    reg=g
                else:
                    term='firstStageRt'
                    reg=g
                
                model=smf.logit(f'correct~{term}',data=reg).fit(disp=0)
                
                if d=='no-up lr':
                    slope_noup.append(model.params[term])
                elif d=='opt-up' or d=='opt-up lr':
                    slope_opt.append(model.params[term])

        ax.plot([0,1],[slope_noup,slope_opt],'.-',color='k',alpha=0.5)
        ax.set_xticks([0,1])
        ax.set_xticklabels([settings.label_map[d] for d in data])

        if stat:
            maxslope=np.max(np.c_[slope_noup,slope_opt])
            _,p=stats.wilcoxon(slope_noup,slope_opt,alternative='less')
            l='*' if p<0.05 else '-'
            label_diff(ax,l,[0,1],maxslope,shrink_x=0.025,base_y=0.05,increment_y=0.05)
            
    def plot_rtAccCohSlope(self,ax,data=['no-up lr','opt-up lr'],remove_up026=False,stat=False):
        """
        slope on first stage RT
        """
        slope_noup=[]
        slope_opt=[]
        
        for subject_name in self.subject_names:
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for d in data:
                g,_,_=self.extractData(d,remove_up026=remove_up026)
            
                g['correct']=g['correct'].astype(int)
                model=smf.logit('correct~firstStageRt+Q("dots.coherence")',data=g).fit(disp=0)
            
                if d=='no-up lr':
                    slope_noup.append(model.params['firstStageRt'])
                elif d=='opt-up' or d=='opt-up lr':
                    slope_opt.append(model.params['firstStageRt'])

        ax.plot([0,1],[slope_noup,slope_opt],'.-',color='k',alpha=0.5)
        ax.set_xticks([0,1])
        ax.set_xticklabels([settings.label_map[d] for d in data])
        
        if stat:
            maxslope=np.max(np.c_[slope_noup,slope_opt])
            _,p=stats.wilcoxon(slope_noup,slope_opt,alternative='less')
            l='*' if p<0.05 else '-'
            label_diff(ax,l,[0,1],maxslope,shrink_x=0.025,base_y=0.05,increment_y=0.05)
    
    def plot_rtAccCohSlope_UR(self,ax,data=['opt-up'],group_RT=False,remove_up026=False,stat=False):
        """
        slope on first stage RT (abolished idea)
        """
        slope_opt=[]
        UR_prop=[]
        
        for subject_name in self.subject_names:
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for d in data:
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                
                g['correct']=g['correct'].astype(int)         
                
                if group_RT:
                    term='_firstStageRtGr5'
                    reg=g.groupby(term)['correct'].mean().reset_index()
                    model=smf.logit(f'correct~{term}',data=reg).fit(disp=0)
                else:
                    term='firstStageRt'
                    reg=g
                    model=smf.logit(f"correct~{term}+Q('dots.coherence')",data=reg).fit(disp=0)
                
                
                slope_opt.append(model.params[term])
                UR_prop.append(g['intoStage2'].sum()/len(g['intoStage2']))
        
        X=sm.add_constant(slope_opt)
        model2=sm.OLS(UR_prop,X).fit(disp=0)
        ax.plot(slope_opt,UR_prop,'.-',color='k')
        ax.plot(slope_opt,model2.predict(X))
        p=model2.params[1]
        l='*' if p<0.05 else '-'
        ax.text((max(slope_opt)-min(slope_opt))/2,max(UR_prop),l,color='k')
    
    def plot_AccUR(self,ax,data=['no-up lr','opt-up lr','opt-up'],remove_up026=False):
        
        cr_std=[]
        cr_opt=[]
        UR_prop=[]
            
        for subject_name in self.subject_names:
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for i,d in enumerate(data):
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                
                if i==0:
                    cr_std.append(g['correct'].mean())
                if i==1:
                    cr_opt.append(g['correct'].mean())
                if i==2:
                    UR_prop.append(g['intoStage2'].sum()/len(g['intoStage2']))
        cr_diff=aa(cr_opt)-aa(cr_std)
        print("pearsonr",stats.pearsonr(cr_diff,UR_prop))
        print("spearmanr",stats.spearmanr(cr_diff,UR_prop))
        model=sm.OLS(UR_prop,sm.add_constant(cr_diff)).fit(disp=0)
        ax.scatter(cr_diff,UR_prop,color='k')
        ax.plot(cr_diff,model.predict(sm.add_constant(cr_diff)),color='k')
        

    def plot_rtVar(self,ax,data=['no-up lr','opt-up lr','opt-up up'],remove_up026=False,stat=False):
        """
        slope on first stage RT
        """
        var_mat=np.zeros((len(self.subject_names),len(data)))
        
        for i,subject_name in enumerate(self.subject_names):
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for j,d in enumerate(data):
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                
                var_mat[i,j]=g['firstStageRt'].var()
                
        idx=np.arange(len(data)).astype(int)
        ax.plot(idx,var_mat.T,'.-',color='k',alpha=0.5)
        ax.set_xticks(idx)
        ax.set_xticklabels([settings.label_map[d] for d in data])
        if stat:
            maxvar=np.max(var_mat)
            xpos=np.arange(len(data))
            _,p01=stats.wilcoxon(var_mat[:,0],var_mat[:,1],alternative='greater')
            _,p02=stats.wilcoxon(var_mat[:,0],var_mat[:,2],alternative='greater')
            _,p12=stats.wilcoxon(var_mat[:,1],var_mat[:,2],alternative='two-sided')
            print(p01,p02,p12)
            l01='*' if p01<0.05 else '-'
            l02='*' if p02<0.05 else '-'
            l12='*' if p12<0.05 else '-'
            label_diff(ax,l01,[0,1],maxvar,shrink_x=0.025,base_y=0.05,increment_y=0.05)
            label_diff(ax,l02,[0,2],maxvar,shrink_x=0.025,base_y=0.2,increment_y=0.05)
            label_diff(ax,l12,[1,2],maxvar,shrink_x=0.025,base_y=0.05,increment_y=0.05)
        
    def plot_RTdiffURprop(self,ax,data=['no-up lr','opt-up lr','opt-up'],remove_up026=False,stat=False):
        """
        RT difference between std and opt is positive realted to UR proportion
        """
        rt_diff=[]
        UR_prop=[]
        for i,subject_name in enumerate(self.subject_names):
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for j,d in enumerate(data):
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                if j==0:
                    rt_base=g['firstStageRt'].to_numpy()
                elif j==1:
                    rt_com=g['firstStageRt'].to_numpy()
                else:
                    UR_prop.append(g['intoStage2'].mean())
            rt_minmax=MinMaxScaler().fit_transform(np.atleast_2d(np.r_[rt_base,rt_com]).T)
            tmp=rt_minmax[:rt_base.shape[0]].mean()-rt_minmax[rt_base.shape[0]:].mean()
            rt_diff.append(tmp)
        
        print("pearsonr",stats.pearsonr(rt_diff,UR_prop))
        print("spearmanr",stats.spearmanr(rt_diff,UR_prop))
        ax.scatter(rt_diff,UR_prop,color='k')
        X=sm.add_constant(rt_diff)
        model=sm.OLS(UR_prop,X).fit(disp=0)
        ax.plot(rt_diff,model.predict(X),color='k')
    
    def plot_SlopediffURprop(self,ax,data=['no-up lr','opt-up lr','opt-up up'],remove_up026=False,stat=False):
        """
        RT difference between std and opt is positive realted to RT-coherence slope
        """
        slope_diff=[]
        UR_prop=[]
        term="Q('dots.coherence')"
        for i,subject_name in enumerate(self.subject_names):
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for j,d in enumerate(data):
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                if j==0:
                    model_slope=smf.ols(f"firstStageRt~{term}",data=g).fit()
                    slope0=model_slope.params[term]
                elif j==1:
                    model_slope=smf.ols(f"firstStageRt~{term}",data=g).fit()
                    slope1=model_slope.params[term]
                else:
                    UR_prop.append(g['intoStage2'].mean())
            tmp=slope0-slope1
            slope_diff.append(tmp)
        ax.scatter(slope_diff,UR_prop,color='k')
        X=sm.add_constant(slope_diff)
        model=sm.OLS(UR_prop,X).fit(disp=0)
        ax.plot(slope_diff,model.predict(X),color='k')

    def plot_SlopediffRTdiff(self,ax,data=['no-up lr','opt-up lr','opt-up up'],remove_up026=False,stat=False):
        """
        RT difference between std and opt is positive realted to RT-coherence slope
        """
        slope_diff=[]
        rt_diff=[]
        term="Q('dots.coherence')"
        for i,subject_name in enumerate(self.subject_names):
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for j,d in enumerate(data):
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                if j==0:
                    model_slope=smf.ols(f"firstStageRt~{term}",data=g).fit()
                    slope0=model_slope.params[term]
                elif j==1:
                    model_slope=smf.ols(f"firstStageRt~{term}",data=g).fit()
                    slope1=model_slope.params[term]
                elif j==2:
                    rt_base=g['firstStageRt'].to_numpy()
                elif j==3:
                    rt_com=g['firstStageRt'].to_numpy()
            slope_diff.append(slope0-slope1)
            rt_minmax=MinMaxScaler().fit_transform(np.atleast_2d(np.r_[rt_base,rt_com]).T)
            tmp=rt_minmax[:rt_base.shape[0]].mean()-rt_minmax[rt_base.shape[0]:].mean()
            rt_diff.append(tmp)
        
        ax.scatter(slope_diff,rt_diff,color='k')
        X=sm.add_constant(slope_diff)
        model=sm.OLS(rt_diff,X).fit(disp=0)
        ax.plot(slope_diff,model.predict(X),color='k')
    # fitted part
    def plot_fittedURAccRTdiff(self,ax,params,y='UR_prop',data=['no-up lr','opt-up lr','opt-up up','opt-up'],remove_up026=False):    
        cr_std=[]
        cr_opt=[]
        rt_diff=[]
        rt_up_diff=[]
        UR_prop=[]
            
        for subject_name in self.subject_names:
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for i,d in enumerate(data):
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                if i==0:
                    cr_std.append(g['correct'].mean())
                    rt_base=g['firstStageRt'].to_numpy()
                if i==1:
                    cr_opt.append(g['correct'].mean())
                    rt_com=g['firstStageRt'].to_numpy()
                if i==2:
                    rt_ur=g['firstStageRt'].to_numpy()
                if i==3:
                    UR_prop.append(g['intoStage2'].sum()/len(g['intoStage2']))
            rt_minmax=MinMaxScaler().fit_transform(np.atleast_2d(np.r_[rt_base,rt_com]).T)
            tmp=rt_minmax[:rt_base.shape[0]].mean()-rt_minmax[rt_base.shape[0]:].mean()
            rt_diff.append(tmp)

            rt_minmax=MinMaxScaler().fit_transform(np.atleast_2d(np.r_[rt_com,rt_ur]).T)
            tmp=rt_minmax[:rt_com.shape[0]].mean()-rt_minmax[rt_ur.shape[0]:].mean()
            rt_up_diff.append(tmp)
        cr_diff=aa(cr_opt)-aa(cr_std)

        if y=='UR_prop':X=UR_prop
        if y=='cr_diff':X=cr_diff
        if y=='rt_up_diff':X=rt_up_diff
        model=sm.OLS(params,sm.add_constant(X)).fit(disp=0)
        ax.scatter(X,params,color='k')
        ax.plot(X,model.predict(sm.add_constant(X)),color='k')
    def plot_Rdiff_avgAccdiff(self,ax,param_R,param_R_std,data=['no-up','opt-up'],remove_up026=False):    
        cr_std=[]
        cr_opt=[]
        R_diff=[]
            
        for subject_name in self.subject_names:
            super(AnalysisData,self).__init__(self.folder,subject_name,self.coherences,self.data_type)

            for i,d in enumerate(data):
                g,_,_=self.extractData(d,remove_up026=remove_up026)
                if i==0:
                    cr_std.append(g['correct'].sum()/g['totalRt'].sum())
                if i==1:
                    cr_opt.append(g['correct'].sum()/g['totalRt'].sum())

        cr_diff=aa(cr_opt)-aa(cr_std)

        R_diff=param_R-param_R_std
        model=sm.OLS(cr_diff,sm.add_constant(R_diff)).fit(disp=0)
        ax.scatter(R_diff,cr_diff,color='k')
        ax.plot(R_diff,model.predict(sm.add_constant(R_diff)),color='k')



def gr_fitted(ax,f):
    """
    generate fitted graph
    
    Parameters
    ----------
    ax : matplotlib.Axes
        figure axes
    f : dict{str,}
        fitted results
    """    
    for k,v in f.items():
        if k[:2]=='y_':
            term=''.join(k.split('_')[1:])
            ax.plot(f['x'],v,
            label=settings.label_map[term],
            color=settings.color_map[term])



   
