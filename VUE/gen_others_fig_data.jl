

# time threshold
using JLD
using NPZ
using VUE

model="2d3b" # tt/2d3b

defs=if model=="tt"
    MarkovDefs1D2BDropout(;
        dt=0.01,
        σ=1.,
        tbar=5.,
        xbar=5.,
        num_states=351,
        mr=3.,
        zs=[-0.26,-0.16,-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26],
        runmed_width=7)
elseif model=="2d3b"
    MarkovDefs2D3B(;
        dt=0.05,
        cov=[1.0 0.0;0.0 1.0],
        tbar=5.,
        x1bar=3.,
        x2bar=3.,
        num_states=[30,30],
        mr=3.,
        zs=[-0.26,-0.16,-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26],
        runmed_width=7)
end

local res_f,modelparameters,modelfit,data_f
if model=="tt"
    res_f="./res_tt/"
    data_f="data_tt"
    modelparameters=Markov1D2BDropoutModelParameters
    modelfit=timeThresholdFit
elseif model=="2d3b"
    res_f="./res_race2D3B"
    data_f="data_2d3b"
    modelparameters=Markov2D3BModelParameters
    modelfit=race2D3BFit
end


subjs=["s01","s02","s03","s06","s08","s09","s10","s12","s13","s15"]
for subj in subjs
    res=load("$res_f/$subj/bps.jld")
    mps=modelparameters(res["best_params"])
    g_upps,g_lows,g_inns=modelfit(mps,defs)

    # calculate accuracy,reaction time and ur proportion
    g_upps_mu=zeros(defs.mid_z)
    g_lows_mu=zeros(defs.mid_z)
    g_inns_mu=zeros(defs.mid_z)
    ur_prop=zeros(defs.mid_z)
    g_acc=zeros(defs.mid_z)
    g_ul_mu=zeros(defs.len_zs)
    g_ur_mu=zeros(defs.len_zs)

    for i in 1:defs.mid_z
        
        sum_g_upp=sum(@view g_upps[:,i])
        sum_g_low=sum(@view g_lows[:,i])
        g_acc[i]=sum_g_upp/(sum_g_upp+sum_g_low)
        
        sum_g_inn=sum(@view g_inns[:,i])
        g_upps_mu[i]=sum(defs.t_vec.*g_upps[:,i])/sum_g_upp  # specific condition 
        g_lows_mu[i]=sum(defs.t_vec.*g_lows[:,i])/sum_g_low 
        g_inns_mu[i]=sum(defs.t_vec.*g_inns[:,i])/sum_g_inn
        ur_prop[i]=sum_g_inn/(sum_g_upp+sum_g_low+sum_g_inn)

        dist=g_upps[:,i].+g_lows[:,i]
        g_ul_mu[defs.mid_z-1+i]=sum(defs.t_vec.*dist)/sum(dist)
        g_ul_mu[defs.mid_z+1-i]=g_ul_mu[defs.mid_z-1+i]
        g_ur_mu[defs.mid_z:end].=g_inns_mu
        g_ur_mu[defs.mid_z:-1:1].=g_inns_mu
    end

    println("time threshold: $subj generated")
    postfix="_rt"
    npzwrite("../connect/$data_f/$subj$postfix.npz",
        Dict(
            "x"=>defs.zs,
            "y_opt-up lr"=>(g_ul_mu.+mps.nondt),
            "y_opt-up up"=>(g_ur_mu.+mps.nondt)
            # "xlabel"=>"Coherence",
            # "ylabel"=>"Reaction time (s)"
        )
    )
    postfix="_ur_prop"
    npzwrite("../connect/$data_f/$subj$postfix.npz",
        Dict(
            "x"=>defs.zs[defs.mid_z:end],
            "y_opt-up"=>ur_prop,
            # "xlabel"=>"|Coherence|",
            # "ylabel"=>"UR proportion"
        )
    )
end