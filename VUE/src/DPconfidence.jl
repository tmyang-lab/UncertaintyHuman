
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
decisionAreaConf(m::ModelParameters,d::Defs;full::Bool=false,ret_value::Bool=false)=begin
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
        V[:,iT],D[:,iT]=max_([
            3*view(Rh_left,:,iT) 0*view(Rh_right,:,iT) 
            2*view(Rh_left,:,iT) 1*view(Rh_right,:,iT) 
            view(EVnext,:,iT).-c*dt])
    end
    if full
        return D,V
    else
        ret_value ? V : D
    end
end