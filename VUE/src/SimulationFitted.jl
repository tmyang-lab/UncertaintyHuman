


vueSim(mps::ModelParameters,d::Defs)=begin
    @unpackStruct mps ModelParameters
    @unpackStruct d Defs

    D=decisionArea(mps,d)
    b=decisionBound(D)
    upp=map(b.upp) do x state2val(x,cr,num_states) end
    low=map(b.low) do x state2val(x,cr,num_states) end
    inn_upp=map(b.inn_upp) do x
        ifelse(x==-1,nothing,state2val(x,cr,num_states))
    end
    inn_low=map(b.inn_low) do x 
        ifelse(x==-1,nothing,state2val(x,cr,num_states))
    end
    c_upps=[]
    c_lows=[]
    c_inns=[]
    for (i_z,z) in enumerate(zs[mid_z:end])
        slope=k*z*t_vec
        c_upp=[]
        c_low=[]
        c_inn=[]
        for i=1:trials_per_z
            noise=accumulate(+,rand(Normal(0,sqrt(σ*dt)),t_N))
            ev=slope+noise
            for t in eachindex(t_vec)
                if ev[t]>upp[t]
                    push!(c_upp,t*dt);break
                elseif ev[t]<low[t]
                    push!(c_low,t*dt);break
                elseif !isnothing(inn_upp[t]) && !isnothing(inn_low[t]) && inn_low[t]<ev[t] && ev[t]<inn_upp[t]
                    push!(c_inn,t*dt);break
                end
            end
        end
        push!(c_upps,c_upp.+nondt)
        push!(c_lows,c_low.+nondt)
        push!(c_inns,c_inn.+nondt)
    end
    return c_upps,c_lows,c_inns
end

timeThresholdSim(mps::Markov1D2BDropoutModelParameters,d::MarkovDefs1D2BDropout;trials_per_z::Int=278)=begin
    @unpackStruct mps Markov1D2BDropoutModelParameters
    @unpackStruct d MarkovDefs1D2BDropout
    # preparation for the stable part
    upp=[b_intercept+b_slope*t for t=t_vec]
    low=[-b_intercept-b_slope*t for t=t_vec]
    dropout_rate=pdf.(truncated(Normal(drop_μ,drop_σ),drop_μ-3*drop_σ,drop_μ+3*drop_σ),t_vec)

    c_upps=[]
    c_lows=[]
    c_inns=[]
    for (i_z,z) in enumerate(zs[mid_z:end])
        slope=k*z*t_vec
        c_upp=[]
        c_low=[]
        c_inn=[]
        for i=1:trials_per_z
            t_b=sample(t_vec,ProbabilityWeights(dropout_rate))
            tidx_b=findmin(abs.(t_b.-t_vec))[2]
            hit=false
            noise=accumulate(+,rand(Normal(0,sqrt(σ*dt)),t_N))
            ev=slope+noise
            for t in eachindex(t_vec[1:tidx_b])
                if ev[t]>upp[t]
                    push!(c_upp,t*dt)
                    hit=true
                    break
                elseif ev[t]<low[t]
                    push!(c_low,t*dt)
                    hit=true
                    break
                end
            end
        
            if !hit
                push!(c_inn,t_b)
            end
        end
        push!(c_upps,c_upp.+nondt)
        push!(c_lows,c_low.+nondt)
        push!(c_inns,c_inn.+nondt)
    end
    return c_upps,c_lows,c_inns
end



race2D3BSim(mps::Markov2D3BModelParameters,d::MarkovDefs2D3B;trials_per_z::Int=278)=begin
    @unpackStruct mps Markov2D3BModelParameters
    @unpackStruct d MarkovDefs2D3B

    c_upps=[]
    c_lows=[]
    c_inns=[]
    for (i_z,z) in enumerate(zs)
        slope=k1*z*t_vec
        ur_slope=k2*(k2_intercept-z)*t_vec
        
        c_upp=[]
        c_low=[]
        c_inn=[]
        for i=1:trials_per_z
            noise=accumulate(+,rand(Normal(0,sqrt(cov[1,1]*dt)),t_N))
            ev=slope+noise

            ur_noise=accumulate(+,rand(Normal(0,sqrt(cov[2,2]*dt)),t_N))
            ur_ev=ur_slope+ur_noise
            for t in eachindex(t_vec)
                if ev[t]>b1[t]
                    push!(c_upp,t*dt)
                    break
                elseif ev[t]<-b1[t]
                    push!(c_low,t*dt)
                    break
                elseif ur_ev[t]>b2
                    push!(c_inn,t*dt)
                    break
                end
            end
        end
        push!(c_upps,c_upp.+nondt)
        push!(c_lows,c_low.+nondt)
        push!(c_inns,c_inn.+nondt)
    end    
    return c_upps,c_lows,c_inns
end