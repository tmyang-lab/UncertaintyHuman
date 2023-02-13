"""
program for fitting response time and ur proportion
using mean value
"""

using VUE
using JLD
using Optim
using Statistics
using Logging,Dates
using UnPack


argsp=ARGS

subjs=["sall","s01","s02","s03","s06","s08","s09","s10","s12","s13","s15"]
weight=parse(Float64,argsp[1]) # weight on reaction time part
method_name=argsp[2] #"NelderMead" #"SimulatedAnnealing"
max_iter=parse(Int64,argsp[3]) # maximum iteration

local x0_init,x0,bestQ
if argsp[4]=="best"
    bestQ=true # using previous best parameters
else
    bestQ=false
    x0_init=parse.(Float64,argsp[4:6]) # '_' is necessary if there are other paprameters 
end

stamp=Dates.format(now(),"yyyymmddHH") # time stamp

for subj in subjs

    println("-------------subject: $subj-------------")
    data_folder="./data/$subj"
    res_folder="./res/$subj"
    if !isdir(res_folder) mkdir(res_folder) end

    ur_jld=load("$data_folder/ur.jld") # read data
    
    subj_stamp=Dates.format(now(),"yyyymmddHH") # time stamp

    ## start logger
    logger_io=open("$res_folder/log_$stamp.txt","w+") # logger file
    logger=SimpleLogger(logger_io)
    with_logger(logger) do 
        @info "$(now()) || $subj" 
    end

    # subject preparation
    @unpack mean_upps,mean_lows,mean_inns,ur_props,accs,n_upps,n_lows,n_inns,n_alls,max_nondt_s=ur_jld

    param_low=[0.,0.,0.,0.] # k,R,EV2,nondt
    param_upp=[50.,50.,50.,max_nondt_s]

    # initial value
    if bestQ # best mode or not
        bp=load("./res/$subj/bps.jld") # k,R,EV_UR,nondt,f
        x0=[bp["best_params"][1:3];max_nondt_s/3.] # k,R,EV_UR,nondt
    else
        x0=[x0_init;max_nondt_s/3. *2.]
    end

    # target function
    local f
    LL(x::Vector{Float64})=begin

        # range check
        if any(x .> param_upp) || any(x .< param_low) 
            with_logger(logger) do 
                @warn("Parameters $x exceed the lower bound $param_low and upper bound $param_upp")
            end
            return NaN
        end 

        # parameters
        m=ModelParameters(x)
        @unpack k,a,EV_UR,nondt=m
        with_logger(logger) do 
            @info "k=$k|a=$a|EV_UR=$(EV_UR)|nondt=$nondt"
        end
        
        # second stage: calculating switching cost
        m2=ModelParameters(k,a,-1.;nondt=nondt)
        EV2=decisionArea(m2,defs2;ret_value=true)[defs2.mid_state,1]
        f=EV2-EV_UR
        if f<0.
            with_logger(logger) do 
                @info "switching cost is negative"
            end
            return NaN  # illegal decision area
        end

        # first stage : decision area
        # D= 1/2 left/right 3 UR 4 waiting
        D=decisionArea(m,defs)
        if !any((@view D[:,end]).==1) || any((@view D[:,1]).==3) || !any(D.==3)
            with_logger(logger) do
                @warn "Decsion area starts with UR or no UR or waiting area"
            end
            return NaN  # illegal decision area
        end

        # first stage : decision bound
        b=decisionBound(D)
        if any(b.low.==-1)
            with_logger(logger) do 
                @warn "Perceptual decision bound is not covering full range"
            end
            return NaN  # illegal decision area
        end
        g_upps,g_lows,g_inns=markovFit(b,m,defs)
        
        # calculate function
        ll=0.
        for i in 1:defs.mid_z
            sum_g_upp=sum(g_upps[:,i])
            sum_g_low=sum(g_lows[:,i])
            g_acc=sum_g_upp/(sum_g_upp+sum_g_low)
            sum_g_inn=sum(g_inns[:,i])
            mean_g_upp=sum(defs.t_vec.*g_upps[:,i])/sum_g_upp  # specific condition 
            mean_g_low=sum(defs.t_vec.*g_lows[:,i])/sum_g_low 
            mean_g_inn=sum(defs.t_vec.*g_inns[:,i])/sum_g_inn
            ur_prop_g=sum_g_inn/(sum_g_upp+sum_g_low+sum_g_inn)

            ll+=weight/n_alls[i]*(
                (n_upps[i]>0.0 ? (n_upps[i] *(mean_g_upp-mean_upps[i]+nondt)^2) : 0.)+
                (n_lows[i]>0.0 ? (n_lows[i] *(mean_g_low-mean_lows[i]+nondt)^2) : 0.)+
                (n_inns[i]>0.0 ? (n_inns[i] *(mean_g_inn-mean_inns[i]+nondt)^2) : 0.))+
                (ur_props[i]>0.0 ? (ur_prop_g-ur_props[i])^2 : 0.)+
                (accs[i]>0.0 ? (g_acc-accs[i])^2 : 0.) # ~with accuracy as target
            # @show i,mean_upps[i],mean_lows[i],mean_inns[i]
            # @show i,mean_g_upp,mean_g_low,mean_g_inn
        end
        with_logger(logger) do 
            @info "Target function: $ll" 
        end

        return ll
    end

    # run optimization
    opt=if method_name=="NelderMead"
        optimize(LL,x0,method=NelderMead(),
        g_tol=1e-6,store_trace=false,show_trace=true,iterations=max_iter)
    elseif method_name=="SimulatedAnnealing"
        optimize(LL,x0,method=SimulatedAnnealing(),
        store_trace=false,show_trace=true,iterations=max_iter)
    else
        throw(DomainError(method_name,"method must be NelderMead or SimulatedAnnealing"))
    end

    # save logger
    flush(logger_io)
    close(logger_io)
    # save result
    save("$res_folder/res_$stamp.jld",
        "time",subj_stamp,
        "method",method_name,
        "minimizer",Optim.minimizer(opt),
        "params",[Optim.minimizer(opt);f], # k,R,EV_UR,nondt,f
        "minimum",Optim.minimum(opt),
        "max_iter",max_iter,
        "weight",weight,
        "x_init",x0)

end
