

subjMean(upp::Dict{Int64, Vector{Float64}},low::Dict{Int64, Vector{Float64}},inn::Dict{Int64, Vector{Float64}};time_scale::Float64=1.)=begin
    #time index
    d_upp=Dict{Int64, Vector{Int64}}()
    d_low=Dict{Int64, Vector{Int64}}()
    d_inn=Dict{Int64, Vector{Int64}}()
    # mean rt    
    mean_upps=zeros(defs.mid_z)
    mean_lows=zeros(defs.mid_z)
    mean_inns=zeros(defs.mid_z)
    ur_props=zeros(defs.mid_z)
    accs=zeros(defs.mid_z)
    # num of trials
    n_upps=zeros(defs.mid_z)
    n_lows=zeros(defs.mid_z)
    n_inns=zeros(defs.mid_z)
    n_alls=zeros(defs.mid_z)

    # mean rt of the subj 
    max_nondt=10.
    real_max_nondt=10.
    for i=1:defs.mid_z
        
        n_upps[i]=length(upp[i])
        n_lows[i]=length(low[i])
        n_inns[i]=length(inn[i])
        n_alls[i]=n_upps[i]+n_lows[i]+n_inns[i]
                
        d_upp[i]=findnearest(upp[i]*time_scale,defs.t_vec)
        d_low[i]=findnearest(low[i]*time_scale,defs.t_vec)
        d_inn[i]=findnearest(inn[i]*time_scale,defs.t_vec)
        
        mean_upps[i]=nanmean(upp[i]*time_scale)
        mean_lows[i]=nanmean(low[i]*time_scale)
        mean_inns[i]=nanmean(inn[i]*time_scale)
        ur_props[i]=n_inns[i]/n_alls[i]
        accs[i]=n_upps[i]/(n_upps[i]+n_lows[i])

        local m
        m=nanminimum([mean_upps[i],mean_lows[i],mean_inns[i]])
        if max_nondt>m max_nondt=m end
        m=nanminimum([upp[i]*time_scale;low[i]*time_scale;inn[i]*time_scale])
        if real_max_nondt>m real_max_nondt=m end
    end

    upplows_all=[];inns_all=[]
    for i=1:defs.mid_z
        upplows_all=vcat(upplows_all,upp[i],low[i])
        inns_all=vcat(inns_all,inn[i])
    end
    mean_upplows_all=mean(upplows_all)
    mean_inns_all=mean(inns_all)
    ur_props_all=sum(n_inns)/sum(n_alls)
    accs_all=sum(n_upps)/sum(vcat(n_upps,n_lows))

    return upp,low,inn,d_upp,d_low,d_inn,mean_upps,mean_lows,mean_inns,ur_props,accs, mean_upplows_all,mean_inns_all,ur_props_all,accs_all,n_upps,n_lows,n_inns,n_alls,max_nondt,real_max_nondt
end

subjMean(upp::Dict{Int64, Vector{Float64}},low::Dict{Int64, Vector{Float64}};time_scale::Float64=1.)=begin
    
    #time index
    d_upp=Dict{Int64, Vector{Int64}}()
    d_low=Dict{Int64, Vector{Int64}}()
    
    mean_upps=zeros(defs.mid_z)
    mean_lows=zeros(defs.mid_z)
    accs=zeros(defs.mid_z)

    n_upps=zeros(defs.mid_z)
    n_lows=zeros(defs.mid_z)
    n_alls=zeros(defs.mid_z)

    # mean rt of the subj 
    max_nondt=10.
    real_max_nondt=10.
    for i=1:defs.mid_z

        n_upps[i]=length(upp[i])
        n_lows[i]=length(low[i])
        n_alls[i]=n_upps[i]+n_lows[i]
        # time index 
        d_upp[i]=findnearest(upp[i],defs.t_vec)
        d_low[i]=findnearest(low[i],defs.t_vec)

        
        mean_upps[i]=nanmean(upp[i]*time_scale)
        mean_lows[i]=nanmean(low[i]*time_scale)
        accs[i]=n_upps[i]/(n_upps[i]+n_lows[i])
        
        local m
        # mean minimum
        m=nanminimum([mean_upps[i],mean_lows[i]])
        if max_nondt>m max_nondt=m end
        m=nanminimum([upp[i]*time_scale;low[i]*time_scale])
        if real_max_nondt>m real_max_nondt=m end
    end

    upplows_all=[]
    for i=1:defs.mid_z
        upplows_all=vcat(upplows_all,upp[i]*time_scale,low[i]*time_scale)
    end
    mean_upplows_all=mean(upplows_all)
    accs_all=sum(n_upps)/sum(vcat(n_upps,n_lows))
    
    @assert max_nondt>0.
    return upp,low,d_upp,d_low,mean_upps,mean_lows,accs, mean_upplows_all,accs_all,n_upps,n_lows,n_alls,max_nondt,real_max_nondt
end