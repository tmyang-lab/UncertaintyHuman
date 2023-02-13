using VUE
using JLD
using Optim
using Statistics
using Logging,Dates
using UnPack


dt=0.05
σ=1.
x1bar=3.
x2bar=3.
cov=[1.0 0.0;0.0 1.0]
zs=[-0.26,-0.16,-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26]
num_states=[30,30]
tbar=5.
mr=3.
runmed_width=7
racedefs=MarkovDefs2D3B(;dt=dt,cov=cov,tbar=tbar,
    x1bar=x1bar,x2bar=x2bar,
    num_states=num_states,mr=mr,zs=zs,runmed_width=runmed_width)

argsp=ARGS

subjs=["s01","s02","s03","s06","s08","s09","s10","s12","s13","s15"]
weight=parse(Float64,argsp[1]) # weight on reaction time part
method_name=argsp[2] #"NelderMead" #"SimulatedAnnealing"
max_iter=parse(Int64,argsp[3]) # maximum iteration

local x0_init,x0,bestQ
if argsp[4]=="best"
    bestQ=true # using previous best parameters
else
    bestQ=false
    x0_init=parse.(Float64,argsp[4:8]) # k1,k2,b1_intercept, b1_slope, b2
end

stamp=Dates.format(now(),"yyyymmddHH") # time stamp

for subj in subjs

    println("-------------subject: $subj-------------")
    data_folder="./data/$subj"
    res_folder="./res_race2D3B/$subj"
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

    param_low=[0.,0.,0.26,0.,0.,0.] # k1,k2,k2_intercept,b1,b2,nondt
    param_upp=[20.,20,3.0,x1bar,x2bar,max_nondt_s]
    
    # initial value
    if bestQ # best mode or not
        bp=load("$(res_folder)/bps.jld") # k1,k2,b1_intercept,b1_slope,b2,nondt
        x0=[bp["best_params"][1:5];max_nondt_s/3.] # k1,k2,b1_intercept,b1_slope,b2,nondt
    else
        x0=[x0_init;max_nondt_s/3. *2.]
    end

    # target function
    LL(x::Vector{Float64})=begin

        # range check

        if any(x .> param_upp) || any(x .< param_low) 
            with_logger(logger) do 
                @warn("Parameters $x exceed the lower bound $param_low and upper bound $param_upp")
            end
            return NaN
        end 

        # parameters
        m=Markov2D3BModelParameters(x)
        @unpack k1,k2,k2_intercept,b1,b2,nondt=m
        with_logger(logger) do 
            @info "k1=$k1,k2=$k2,k2_intercept=$k2_intercept,b1=$b1,b2=$b2,nondt=$nondt"
        end
        
        g_upps,g_lows,g_inns=race2D3BFit(m,racedefs)
        
        # calculate function
        ll=0.
        for i in 1:racedefs.mid_z
            sum_g_upp=sum(g_upps[:,i])
            sum_g_low=sum(g_lows[:,i])
            g_acc=sum_g_upp/(sum_g_upp+sum_g_low)
            sum_g_inn=sum(g_inns[:,i])
            mean_g_upp=sum(racedefs.t_vec.*g_upps[:,i])/sum_g_upp  # specific condition 
            mean_g_low=sum(racedefs.t_vec.*g_lows[:,i])/sum_g_low 
            mean_g_inn=sum(racedefs.t_vec.*g_inns[:,i])/sum_g_inn
            ur_prop_g=sum_g_inn/(sum_g_upp+sum_g_low+sum_g_inn)

            ll+=weight/n_alls[i]*(
                (n_upps[i]>0 ? (n_upps[i] *(mean_g_upp-mean_upps[i]+nondt)^2) : 0.)+
                (n_lows[i]>0 ? (n_lows[i] *(mean_g_low-mean_lows[i]+nondt)^2) : 0.)+
                (n_inns[i]>0 ? (n_inns[i] *(mean_g_inn-mean_inns[i]+nondt)^2) : 0.))+
                (ur_prop_g-ur_props[i])^2+(g_acc-accs[i])^2 # ~with accuracy as target
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
        "params",Optim.minimizer(opt), # k1,k2,k2_intercept,b1,b2,nondt
        "minimum",Optim.minimum(opt),
        "max_iter",max_iter,
        "weight",weight,
        "x_init",x0)

end
