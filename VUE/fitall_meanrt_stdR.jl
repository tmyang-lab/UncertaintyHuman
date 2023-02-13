"""
program for fitting response time and ur proportion
"""

using VUE
using JLD
using Optim
using Statistics
using Logging,Dates
using UnPack


println("Arguments $ARGS")


subjs=["s01","s02","s03","s06","s08","s09","s10","s12","s13","s15"]

weight=parse(Float64,ARGS[1]) # weight=1.
method_name=ARGS[2] #"NelderMead" #"SimulatedAnnealing"
max_iter=parse(Int64,ARGS[3])   

local x0_init,x0,bestQ
if ARGS[4]=="best"
    bestQ=true
else
    bestQ=false
    x0_init=parse.(Float64,ARGS[4:4]) # '_' is necessary if there are other paprameters 
end

stamp=Dates.format(now(),"yyyymmddHH") # time stamp

for subj in subjs

    println("-------------subject: $subj-------------")
    data_folder="./data/$subj"
    res_folder="./res/$subj"
    if !isdir(res_folder) mkdir(res_folder) end
    nour_jld=load("$data_folder/nour.jld") # read data
    k,_,_,nondt,_=load("$res_folder/bps.jld")["best_params"] # k,R,EV_UR,nondt,f
    @show k,nondt
    subj_stamp=Dates.format(now(),"yyyymmddHH") # time stamp

    ## start logger
    logger_io=open("$res_folder/log_std_$stamp.txt","w+") # logger file
    logger=SimpleLogger(logger_io)
    with_logger(logger) do 
        @info "$(now()) || $subj" 
    end

    # subject preparation
    @unpack mean_upps_std,mean_lows_std,accs_std,n_upps_std,n_lows_std,n_alls_std,max_nondt_s=nour_jld

    param_low=[0.] # R
    param_upp=[50.] 

    # initial value
    x0=if bestQ # best mode or not
        bp=load("$res_folder/bps.jld") # k,R,nondt
        x0=[bp["best_params"][2]] # R
    else
        x0=[x0_init]
    end
    @show x0

    # target function
    LL(x::Vector{Float64})=begin

        # range check
        if any(x .> param_upp) || any(x .< param_low) 
            with_logger(logger) do 
                @warn("Parameters $k exceed the lower bound $param_low and upper bound $param_upp")
            end
            return NaN
        end

        # parameters
        a=x[1]
        m=ModelParameters(k,a,-Inf64;nondt=nondt)
        with_logger(logger) do 
            @info "k=$k|a=$a|nondt=$nondt"
        end
        
        # first stage : decision area
        D=decisionArea(m,defs)
        if !any((@view D[:,end]).==1) || !any((@view D[:,1]).==4) 
            with_logger(logger) do
                @warn "Decsion area starts without waiting area or ends without perceptual decisions."
            end
            return NaN  # illegal decision area
        end

        # first stage : decision bound
        b=decisionBound(D);
        if any(b.low.==-1)
            with_logger(logger) do 
                @warn "Perceptual decision bound is not covering full range"
            end
            return NaN  # illegal decision area
        end
        g_upps,g_lows,_=markovFit(b,m,defs)

        # calculate function
        ll=0.
        for i in 1:defs.mid_z
            sum_g_upp=sum(g_upps[:,i])
            sum_g_low=sum(g_lows[:,i])
            g_acc=sum_g_upp/(sum_g_upp+sum_g_low)

            mean_g_upp=sum(defs.t_vec.*g_upps[:,i]./sum_g_upp)  # specific condition 
            mean_g_low=sum(defs.t_vec.*g_lows[:,i]./sum_g_low) 

            ll+=weight/n_alls_std[i]*(
                (n_upps_std[i]>0. ? n_upps_std[i]*(mean_g_upp-mean_upps_std[i]+nondt)^2 : 0.)
                +
                (n_lows_std[i]>0. ? n_lows_std[i]*(mean_g_low-mean_lows_std[i]+nondt)^2 : 0.))
                +(g_acc-accs_std[i])^2 # ~with accuracy as target
        end
        with_logger(logger) do 
            @info "Target function: $ll" 
        end

        return ll
    end

    # run optimization
    opt=if method_name=="NelderMead"
        throw(DomainError(method_name,"NelderMead method is not available for single variable"))
    elseif method_name=="SimulatedAnnealing"
        optimize(LL,x0,method=SimulatedAnnealing(),
        store_trace=false,show_trace=true,iterations=max_iter)
    else
        throw(DomainError(method_name,"Method must be NelderMead or SimulatedAnnealing"))
    end

    # save logger
    flush(logger_io)
    close(logger_io)
    # save result
    save("$res_folder/res_std_$stamp.jld",
        "time",subj_stamp,
        "method",method_name,
        "minimizer",Optim.minimizer(opt),
        "params",[k,Optim.minimizer(opt)[1],nondt], # k,R,EV_UR,nondt,f
        "minimum",Optim.minimum(opt),
        "max_iter",max_iter,
        "weight",weight,
        "x_init",x0)

end
