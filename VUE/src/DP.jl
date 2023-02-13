"""
Dynamic programming part
"""


"""
cols are possible choices
"""
@inline function max_(v::Matrix{Float64})
    m,idx=findmax(v,dims=2)
    Dt=vec(getindex.(idx,2))
    # Dt[(Dt.==1) .&& (v[1].==v[2])]=12
    return m,Dt
end

"""
normalized normal distribution
"""

pdf_(x::Vector{Float64},mu::Float64,sigma::Float64)=begin
    val=exp.(-((x .-mu)./sigma).^2 ./2.)
    return val/sum(val)
end

npwhere(z::BitMatrix)=begin
    m=findall(z)
    return map(z->z[1],m),map(z->z[2],m)
end

struct BoundPositions
    upp::Vector{Int64}
    low::Vector{Int64}
    inn_upp::Vector{Int64}
    inn_low::Vector{Int64}
    BoundPositions(upp::Vector{Int64},low::Vector{Int64},inn_upp::Vector{Int64},inn_low::Vector{Int64})=begin
        new(upp,low,inn_upp,inn_low)
    end
end

"""
model for fitting
"""
decisionArea(m::ModelParameters,d::Defs;full::Bool=false,ret_value::Bool=false)=begin
    c=1. # cost is set to be constant
    
    @unpackStruct d Defs
    @unpackStruct m ModelParameters
    
    EV_UR_vec=fill(EV_UR,num_states)
    # big Z part
    prob_z=zeros(size(x_grids)...,len_zs)
    C=k/(2*var)
    x2_grids=2 .* x_grids
    kt_grids=k .* t_grids
    @inbounds for r=1:len_zs 
        for i=1:len_zs
            if i!=r
                prob_z[:,:,r].+=ps[i]/ps[r] .* exp.(C*(zs[i]-zs[r]).*(x2_grids.-kt_grids.*(zs[i]+zs[r])))
            end
        end
        prob_z[:,:,r].=1 ./(1 .+ view(prob_z,:,:,r))
    end

    # coherence always contain zero
    if isodd(len_zs)
        prob_left=sum(view(prob_z,:,:,1:mid_z-1);dims=3)[:,:,1] .+ view(prob_z,:,:,mid_z) ./ 2
    else
        prob_left=sum(view(prob_z,:,:,1:mid_z-1);dims=3)[:,:,1]
    end
    
    # prob dx under z
    l_dx=length(dx_vec)
    prob_dx_under_z=zeros(l_dx,len_zs)
    @inbounds for i=1:len_zs
        prob_dx_under_z[:,i].=pdf_(dx_vec,k*zs[i]*dt,sqrt(dt)*σ)
    end

    prob_dx_under_x=zeros(size(x_grids)...,l_dx)
    @inbounds for i=1:l_dx
        for j=1:len_zs
            prob_dx_under_x[:,:,i].+=prob_dx_under_z[i,j] .* view(prob_z,:,:,j)
        end
    end
    
    Rh_left=prob_left .* a
    Rh_right=(1 .- prob_left) .* a

    V=zeros(num_states,t_N)
    D=zeros(Int64,num_states,t_N)
    EVnext=zero(V)
    EVnext[:,end].=-Inf

    V[:,end],D[:,end]=max_([(@view Rh_left[:,end]) (@view Rh_right[:,end]) EV_UR_vec])
    
    @inbounds for iT=t_N-1:-1:1
        for (i,x) in enumerate(eachrow(xp1_state)) # expectation
            EVnext[i,iT]=sum(V[x,iT+1] .* view(prob_dx_under_x,i,iT,:))
        end
        V[:,iT],D[:,iT]=max_([view(Rh_left,:,iT) view(Rh_right,:,iT) EV_UR_vec view(EVnext,:,iT).-c*dt])
    end
    if full
        return D,V
    else
        ret_value ? V : D
    end
end


"""
    fixRTs different fix RTs, from small to large
    pFixRTs probability on theres fix RTs
    tbar should be same fixRTs[end]
    
    to serve Kiani result
"""
decisionAreaFixed(m::ModelParameters,d::Defs,
    fix_RTs::Vector{Float64},p_fix_RTs::Vector{Float64},
    ;prob_sr=0.5,full::Bool=false)=begin
    # probabiltiy of sure target
    c=1.
    @unpackStruct d Defs
    @unpackStruct m ModelParameters
    
    # prepare fix duration #TODO integrate into defs in the future
    t_NFix=length(fix_RTs)
    idx_fixRTs = Int64.(fix_RTs .÷ dt)
    prob_nextFixRT=zeros(t_NFix)
    
    for i=1:t_NFix-1    
        prob_nextFixRT[i]=sum(p_fix_RTs[i+1:end])/sum(p_fix_RTs[i:end])
    end
    # end preparation of fix duration

    EV_UR_vec=fill(EV_UR,num_states)
    # big Z part
    prob_z=zeros(size(x_grids)...,len_zs)
    C=k/(2*var)
    x2_grids=2 .* x_grids
    kt_grids=k .* t_grids
    for r=1:len_zs
        for i=1:len_zs
            if i!=r
                prob_z[:,:,r].+=ps[i]/ps[r] .* exp.(C*(zs[i]-zs[r]).*(x2_grids.-kt_grids.*(zs[i]+zs[r])))
            end
        end
        prob_z[:,:,r].=1 ./(1 .+ view(prob_z,:,:,r))
    end

    # coherence always contain zero
    prob_left=sum(view(prob_z,:,:,1:mid_z-1);dims=3)[:,:,1] .+ view(prob_z,:,:,mid_z) ./ 2


    # prob dx under z
    l_dx=length(dx_vec)
    prob_dx_under_z=zeros(l_dx,len_zs)
    for i=1:len_zs
        prob_dx_under_z[:,i].=pdf_(dx_vec,k*zs[i]*dt,sqrt(dt)*σ)
    end

    # TODO: prob dx under x # the largest time consumuing part for now
    prob_dx_under_x=zeros(size(x_grids)...,l_dx)
    for i=1:l_dx, j=1:len_zs
        prob_dx_under_x[:,:,i].+=prob_dx_under_z[i,j] .* view(prob_z,:,:,j)
    end
    
    Rh_left=prob_left .* a
    Rh_right=(1 .- prob_left) .* a

    V=zeros(num_states,t_N)
    D=zeros(Int64,num_states,t_N)

    V_fix=zeros(num_states,t_NFix)
    D_fix=zeros(Int64,num_states,t_NFix)

    VUR_fix=zeros(num_states,t_NFix)
    DUR_fix=zeros(Int64,num_states,t_NFix)


    EVnext=zero(V)
    EVnext[:,end].=-Inf
    NOUR_vec=fill(-Inf,num_states)
    VURend,_=max_([(@view Rh_left[:,end]) (@view Rh_right[:,end]) EV_UR_vec])
    Vend,_=max_([(@view Rh_left[:,end]) (@view Rh_right[:,end])])
    V[:,end]=prob_sr*VURend+(1-prob_sr)*Vend
    VUR_fix[:,end],DUR_fix[:,end]=max_([(@view Rh_left[:,end]) (@view Rh_right[:,end]) EV_UR_vec])
    V_fix[:,end],D_fix[:,end]=max_([(@view Rh_left[:,end]) (@view Rh_right[:,end]) NOUR_vec])

    iTFix=t_NFix-1
    local t_FixQ

    for iT=t_N-1:-1:1
        
        if iTFix>0 
            t_FixQ=iT==idx_fixRTs[iTFix]
        end
        if t_FixQ
            VUR_fix[:,iTFix],DUR_fix[:,iTFix]=max_([(@view Rh_left[:,iT]) (@view Rh_right[:,iT]) EV_UR_vec])
            V_fix[:,iTFix],D_fix[:,iTFix]=max_([(@view Rh_left[:,iT+1]) (@view Rh_right[:,iT+1]) NOUR_vec])
        end
        for (i,x) in enumerate(eachrow(xp1_state)) # expectation

            if t_FixQ # at the time bound
                EVnext[i,iT]=prob_nextFixRT[iTFix]*sum(V[x,iT+1] .* view(prob_dx_under_x,i,iT,:))+
                (1 .- prob_nextFixRT[iTFix])*V_fix[i,iTFix] # no decision + forced decision
            else
                EVnext[i,iT]=sum(V[x,iT+1] .* view(prob_dx_under_x,i,iT,:))
            end
        end

        V[:,iT],D[:,iT]=max_([NOUR_vec view(EVnext,:,iT).-c*dt])
        
        if t_FixQ 
            iTFix-=1
            t_FixQ=false
        end
    end

    if full
        return V,D_fix,V_fix,DUR_fix,VUR_fix
    else
        return D_fix,DUR_fix
    end
end







decisionBound(D::Matrix{Int64})=begin
    # transfer D to bound
    t_N=size(D,2)
    upp=-ones(Int64,t_N) # TODO: the elemenet is -1
    low=-ones(Int64,t_N)
    inn_upp=-ones(Int64,t_N)
    inn_low=-ones(Int64,t_N)

    waitp=(@view D[2:end,:]).==4
    waitm=(@view D[1:end-1,:]).==4
    rightp=(@view D[2:end,:]).==2
    leftm=(@view D[1:end-1,:]).==1

    val,idx=npwhere(waitp .& leftm)
    low[idx].=val

    val,idx=npwhere(waitm .& rightp)
    upp[idx].=val .+ 1

    val,idx=npwhere(((@view D[2:end,:]).==3) .& (waitm .| leftm))
    inn_low[idx].=val .+ 1

    val,idx=npwhere(((@view D[1:end-1,:]).==3) .& (waitp .| rightp))
    inn_upp[idx].=val

    low[end]=low[end-1] # fill the last term
    upp[end]=upp[end-1] 
    inn_low[end]=inn_low[end-1]
    inn_upp[end]=inn_upp[end-1]
    # transfer bound to rt distribution
    return BoundPositions(upp,low,inn_upp,inn_low)
end


include("AuxMarkovFit.jl")
include("MarkovFit.jl")
include("MarkovFitOthers.jl")