"""
generate julia-compitable data for fitting from python project
mean reaction time and other properties are given
"""

using DataFrames
using CSV,JLD
using Statistics
using VUE
using NaNStatistics

subjs=["s01","s02","s03","s06","s08","s09","s10","s12","s13","s15"]
num_subjs=length(subjs)

cohs=[0.0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26]
num_cohs=length(cohs)
cohToIdx(xx_coh::Float64)=findall(x->x==xx_coh,cohs)[1]
getRT(df::DataFrame)=Dict(i=>df[df[!,:coh].==i,:firstStageRt] for i in eachindex(cohs))
getUR(df::DataFrame)=Dict(i=>mean(df[df[!,:coh].==i,:intoStage2]) for i in eachindex(cohs))

sall_mean_upps=zeros(num_subjs,num_cohs)
sall_mean_lows=zeros(num_subjs,num_cohs)
sall_mean_inns=zeros(num_subjs,num_cohs)
sall_ur_props=zeros(num_subjs,num_cohs)
sall_accs=zeros(num_subjs,num_cohs)
sall_n_upps=zero(cohs)
sall_n_lows=zero(cohs)
sall_n_inns=zero(cohs)
sall_n_alls=zero(cohs)
sall_max_nondt=10.


sall_mean_upps_std=zeros(num_subjs,num_cohs)
sall_mean_lows_std=zeros(num_subjs,num_cohs)
sall_ur_props_std=zeros(num_subjs,num_cohs)
sall_accs_std=zeros(num_subjs,num_cohs)
sall_n_upps_std=zero(cohs)
sall_n_lows_std=zero(cohs)
sall_n_inns_std=zero(cohs)
sall_n_alls_std=zero(cohs)
sall_max_nondt_std=10.

sall_max_nondt_s=0.

for (i,subj) in enumerate(subjs)

    println("--------subject:$subj---------")
    df=CSV.read("../subj_csv_data/formodel/$subj.csv",DataFrame)

    # filter data
    df=df[df[!,:firstStageRt].>0.25,:]
    ## correct choices
    df[!,:upp]=((df[!,"dots.dircoherence"].>=0) .& (df[!,:choosedDirection].=="right")) .| ((df[!,"dots.dircoherence"].<0) .& (df[!,:choosedDirection].=="left"))
    df[!,:coh]=df[!,"dots.dircoherence"] .|> abs .|> cohToIdx

     
    dfur=df[df[!,:trialType].=="opt-up",:]
    @show minimum(dfur[!,:firstStageRt])
    dfnour=df[df[!,:trialType].=="no-up",:];
    @show minimum(dfnour[!,:firstStageRt])

    data_folder="./data/$subj"
    if !isdir(data_folder) mkdir(data_folder) end
    res_folder="./res/$subj"
    if !isdir(res_folder) mkdir(res_folder) end

    
    q_inn=dfur[!,:intoStage2] # into stage 2
    q_upp=dfur[!,:upp] # correct choices
    q_upp_nour=dfnour[!,:upp] # incorrect choices
    
    # reaction time extraction
    inn=getRT(dfur[q_inn,:])
    upp=getRT(dfur[q_upp .& .!q_inn,:])
    low=getRT(dfur[.!q_upp .& .!q_inn,:])

    upp_std=getRT(dfnour[q_upp_nour,:])
    low_std=getRT(dfnour[.!q_upp_nour,:])

    # CUE task
    upp,low,inn,d_upp,d_low,d_inn,
    mean_upps,mean_lows,mean_inns,ur_props,accs, 
    mean_upplows_all,mean_inns_all,ur_props_all,accs_all,
    n_upps,n_lows,n_inns,n_alls,max_nondt,real_max_nondt=subjMean(upp,low,inn)
    
    # STD task
    upp_std,low_std,d_upp_std,d_low_std,mean_upps_std,mean_lows_std,accs_std, 
    mean_upplows_all_std,accs_all_std,
    n_upps_std,n_lows_std,n_alls_std,max_nondt_std,real_max_nondt_std=subjMean(upp_std,low_std)    


    max_nondt_s=minimum([real_max_nondt,real_max_nondt_std])
    @show max_nondt_s
    @assert max_nondt_s>0.
    save("$data_folder/ur.jld",        
        "upp",upp,
        "low",low,
        "inn",inn,
        "d_upp",d_upp,
        "d_low",d_low,
        "d_inn",d_inn,
        "mean_upps",mean_upps,
        "mean_lows",mean_lows,
        "mean_inns",mean_inns,
        "ur_props",ur_props,
        "accs",accs, 
        "mean_upplows_all",mean_upplows_all,
        "mean_inns_all",mean_inns_all,
        "ur_props_all",ur_props_all,
        "accs_all",accs_all,
        "n_upps",n_upps,
        "n_lows",n_lows,
        "n_inns",n_inns,
        "n_alls",n_alls,
        "max_nondt",max_nondt,
        "max_nondt_s",max_nondt_s # all minimum availabel reaction time
        )
    save("$data_folder/nour.jld",
        "upp_std",upp_std,
        "low_std",low_std,
        "d_upp_std",d_upp_std,
        "d_low_std",d_low_std,
        "mean_upps_std",mean_upps_std,
        "mean_lows_std",mean_lows_std,
        "accs_std",accs_std, 
        "mean_upplows_all_std",mean_upplows_all_std,
        "accs_all_std",accs_all_std,
        "n_upps_std",n_upps_std,
        "n_lows_std",n_lows_std,
        "n_alls_std",n_alls_std,
        "max_nondt_std",max_nondt_std,
        "max_nondt_s",max_nondt_s
        )

    # optional assisstance task
    sall_mean_upps[i,:]=mean_upps
    sall_mean_lows[i,:]=mean_lows
    sall_mean_inns[i,:]=mean_inns
    sall_ur_props[i,:]=ur_props
    sall_accs[i,:].+=accs
    sall_n_upps.+= replace(n_upps,NaN=>0.) 
    sall_n_lows.+= replace(n_lows,NaN=>0.) 
    sall_n_inns.+= replace(n_inns,NaN=>0.)

    global sall_max_nondt
    sall_max_nondt=sall_max_nondt>max_nondt ? max_nondt : sall_max_nondt
    
    # standard task
    sall_mean_upps_std[i,:]=mean_upps_std
    sall_mean_lows_std[i,:]=mean_lows_std
    sall_accs_std[i,:]=accs_std
    sall_n_upps_std.+= replace(n_upps_std, NaN=>0.)
    sall_n_lows_std.+= replace(n_lows_std,NaN=>0.) 

    global sall_max_nondt_std
    sall_max_nondt_std=sall_max_nondt_std>max_nondt_std ? max_nondt_std : sall_max_nondt_std


    # all reaction time
    global sall_max_nondt_s
    sall_max_nondt_s+=max_nondt_s
end


sall_n_alls .= sall_n_upps .+ sall_n_lows .+ sall_n_inns
sall_max_nondt_s/=num_subjs

data_folder="./data/sall"
if !isdir(data_folder) mkdir(data_folder) end
res_folder="./res/sall"
if !isdir(res_folder) mkdir(res_folder) end


save("$data_folder/ur.jld",        
        "mean_upps",vec(nanmean(sall_mean_upps;dims=1)),
        "mean_lows",vec(nanmean(sall_mean_lows;dims=1)),
        "mean_inns",vec(nanmean(sall_mean_inns;dims=1)),
        "ur_props",vec(nanmean(sall_ur_props;dims=1)),
        "accs",vec(nanmean(sall_accs;dims=1)), 
        
        "mean_upplows_all",[],
        "mean_inns_all",[],
        "ur_props_all",[],
        "accs_all",[],

        "n_upps",sall_n_upps,
        "n_lows",sall_n_lows,
        "n_inns",sall_n_inns,
        "n_alls",sall_n_alls,
        "max_nondt",sall_max_nondt,
        "max_nondt_s",sall_max_nondt_s
        )


sall_n_alls_std .= sall_n_upps_std .+ sall_n_lows_std .+ sall_n_inns_std

data_folder="./data/sall"
if !isdir(data_folder) mkdir(data_folder) end
save("$data_folder/nour.jld",
        "mean_upps_std",vec(nanmean(sall_mean_upps_std;dims=1)),
        "mean_lows_std",vec(nanmean(sall_mean_lows_std;dims=1)),
        "accs_std",vec(nanmean(sall_accs_std;dims=1)), 
        
        "mean_upplows_all_std",[],
        "accs_all_std",[],
        
        "n_upps_std",sall_n_upps_std,
        "n_lows_std",sall_n_lows_std,
        "n_alls_std",sall_n_alls_std,
        "max_nondt_std",sall_max_nondt_std,
        "max_nondt_s",sall_max_nondt_s
        )