"""
return values positions in vec
"""
findnearest(values::Vector{T},vec::Vector{T}) where T<:AbstractFloat=begin
    m=length(values)    
    indices=zeros(Int64,m)
    for i in 1:m
        indices[i]=findmin(abs.(vec .- values[i]))[2]
    end
    return indices
end

"""
Parameter part
values are to be find
return values position in vec
"""
findnearest(values::Matrix{T},vec::Vector{T}) where T<:AbstractFloat=begin
    m,n=size(values)    
    indices=zeros(Int64,m,n)
    for i in 1:m, j in 1:n
        indices[i,j]=findmin(abs.(vec .- values[i,j]))[2]
    end
    return indices
end

struct Defs
    #___control part___#
    dt::Float64
    σ::Float64
    tw1::Float64
    tw2::Float64
    
    tbar::Float64
    xbar::Float64
    num_states::Int64
    mr::Float64 # mapping range
    
    zs::Vector{Float64}
    ps::Vector{Float64}

    ## markov fit
    runmed_width::Int64 # even
    runmed_width_inn::Int64 # even
    
    #___generated part___#
    var::Float64
    t_vec::Vector{Float64}
    t_N::Int64 # the length of t vec
    z_max::Float64
    mid_z::Int64
    mid_state::Int64
    sigma_range::Float64

    dx::Float64
    dx_vec::Vector{Float64}
    x_vec::Vector{Float64}
    x_grids::Matrix{Float64}
    t_grids::Matrix{Float64}
    xp1_mat::Matrix{Float64}
    xp1_state::Matrix{Int64}

    len_zs::Int64

    ## markov fit
    cr::Vector{Float64} # covered range

    function Defs(;dt::Float64,σ::Float64,tw1::Float64,tw2::Float64,tbar::Float64,xbar::Float64,num_states::Int64,mr::Float64,zs::Vector{Float64},ps::Vector{Float64},runmed_width::Int64,runmed_width_inn::Int64)
        runmed_width!=0 && iseven(runmed_width) && throw(DomainError(runmed_width,"must be an odd number"))
        runmed_width_inn!=0 && iseven(runmed_width_inn) && throw(DomainError(runmed_width_inn,"must be an odd number"))

        ps=ps./sum(ps)
        
        var=σ^2.
        t_vec=collect(dt:dt:tbar)
        t_N=length(t_vec)
        z_max=maximum(zs)
        mid_z=length(zs)÷2+1 # ceil of num_states/2
        mid_state=num_states÷2+1 # ceil of num_states/2
        sigma_range=σ*mr
        
        dx=(xbar-(-xbar))/(num_states-1)
        dx_vec=[reverse(collect(0:-dx:-sigma_range));collect(0:dx:sigma_range)[2:end]]
        x_vec=[reverse(collect(0:-dx:-xbar));collect(0:dx:xbar)[2:end]]
        x_grids,t_grids=ndgrid(x_vec,t_vec)

        xp1_mat=x_vec .+ dx_vec'
        xp1_state=findnearest(xp1_mat,x_vec)

        len_zs=length(zs)

        ## markov fit
        cr=[-xbar,xbar]
        
        new(dt,σ,tw1,tw2,tbar,xbar,num_states,mr,zs,ps,runmed_width,runmed_width_inn,
        var,t_vec,t_N,z_max,mid_z,mid_state,sigma_range,dx,dx_vec,x_vec,x_grids,t_grids,xp1_mat,xp1_state,len_zs,cr)
    end
end

# first stage
defs=Defs(;dt=0.02,σ=1.0,tw1=1.0,tw2=1.0,tbar=10.0,xbar=5.0,num_states=351,mr=3.0,
    zs=[-0.26,-0.16,-0.12,-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.26],
    ps=[1.,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1],
    runmed_width=7,runmed_width_inn=7)
# second stage
defs2=Defs(;dt=0.02,σ=1.0,tw1=1.0,tw2=1.0,tbar=10.0,xbar=5.0,num_states=351,mr=3.0,
    zs=[-0.26,0.26],
    ps=[1.,1],
    runmed_width=7,runmed_width_inn=7)


"""
q : concrete struct
s : struct type name
"""
macro unpackStruct(q::Symbol,s::Symbol)
    code =  Expr(:block, [ :($field = $q.$field) for field in fieldnames(eval(s)) ]...)
    esc(code)
end

struct ModelParameters
    k::Float64
    a::Float64
    EV_UR::Float64
    nondt::Float64
    T0sigma::Float64
    ModelParameters(k::T,a::T,EV_UR::T,nondt::T,T0sigma::T) where T<:AbstractFloat = begin
        new(k,a,EV_UR,nondt,T0sigma)
    end
    ModelParameters(k::T,a::T,EV_UR::T;nondt::T=0.,T0sigma::T=0.) where T<:AbstractFloat = begin
        new(k,a,EV_UR,nondt,T0sigma)
    end
    ModelParameters(x::AbstractVector{T}) where T<:AbstractFloat = begin
        if length(x)==3
            ModelParameters(x[1],x[2],x[3])
        elseif length(x)==4
            ModelParameters(x[1],x[2],x[3];nondt=x[4])
        else
            new(x...)
        end
    end
end


## 1d2b dropout
val2state(val::Float64,cr::AbstractVector{Float64},num_states::Int)=begin
    temp=(val-cr[1]+(cr[2]-cr[1])/(2*num_states))/(cr[2]-cr[1])*num_states
    temp<.5 && (temp=0)
    temp>num_states+0.5 && (temp=num_states+1)
    state=round(Int,temp)
    return state
end

struct MarkovDefs1D2BDropout
    #___control part___#
    dt::Float64
    σ::Float64
    
    tbar::Float64
    xbar::Float64
    num_states::Int64
    mr::Float64 # mapping range
    zs::Vector{Float64}
    runmed_width::Int64 # even

    ## generated part
    t_vec::Vector{Float64}
    t_N::Int64
    mid_z::Int64
    init_state::Int64
    cr::Vector{Float64} # covered range
    len_zs::Int64


    function MarkovDefs1D2BDropout(;dt::Float64,σ::Float64,
        tbar::Float64,xbar::Float64,
        num_states::Int64,mr::Float64,zs::AbstractVector{Float64},runmed_width::Int64)

        runmed_width!=0 && iseven(runmed_width) && throw(DomainError(runmed_width,"must be an odd number"))

        t_vec=collect(dt:dt:tbar)
        t_N=length(t_vec)
        mid_z=ceil(Int,length(zs)/2) # ceil of num_states/2
        cr=[-xbar,xbar]
        init_state=val2state(0.,cr,num_states)
        len_zs=length(zs)

        new(dt,σ,tbar,xbar,num_states,mr,zs,runmed_width,
        t_vec,t_N,mid_z,init_state,cr,len_zs)
    end
end

struct Markov1D2BDropoutModelParameters
    k::Float64
    b_intercept::Float64
    b_slope::Float64
    drop_μ::Float64
    drop_σ::Float64
    nondt::Float64
    T0sigma::Float64
    Markov1D2BDropoutModelParameters(k::T,b_intercept::T,b_slope::T,drop_μ::T,drop_σ::T,nondt::T,T0sigma::T) where T<:AbstractFloat = begin
        new(k,b_intercept,b_slope,drop_μ,drop_σ,nondt,T0sigma)
    end
    Markov1D2BDropoutModelParameters(k::T,b_intercept::T,b_slope::T,drop_μ::T,drop_σ::T;nondt::T=0.,T0sigma::T=0.) where T<:AbstractFloat = begin
        new(k,b_intercept,b_slope,drop_μ,drop_σ,nondt,T0sigma)
    end
    Markov1D2BDropoutModelParameters(x::AbstractVector{T}) where T<:AbstractFloat = begin
        if length(x)==5
            Markov1D2BDropoutModelParameters(x[1:5]...)
        elseif length(x)==6
            Markov1D2BDropoutModelParameters(x[1:5]...;nondt=x[6])
        else
            new(x...)
        end
    end
end


## 2d 3b
val2state(val::AbstractVector{Float64},cr::AbstractMatrix{Float64},num_states::AbstractVector{Int})=begin
    d1=cr[1,2]-cr[1,1]
    d2=cr[2,2]-cr[2,1]
    col=(val[1]-cr[1,1]+d1/(2*num_states[1]))/d1*num_states[1]
    row=(val[2]-cr[2,1]+d2/(2*num_states[2]))/d2*num_states[2]
    (col<0.5 || col>num_states[1]+0.5) && (col=0)
    (row<0.5 || row>num_states[2]+0.5) && (row=0)
    col=round(col)
    row=round(row)
    state = Int(ifelse(col==0 || row==0,0,(row.-1)*num_states[1].+col))
    return state
end
struct MarkovDefs2D3B
    #___control part___#
    dt::Float64
    cov::Matrix{Float64}
    
    tbar::Float64
    x1bar::Float64
    x2bar::Float64
    num_states::Vector{Int64}
    
    mr::Float64 # mapping range
    zs::Vector{Float64}
    
    runmed_width::Int64 # even

    ## generated part
    t_vec::Vector{Float64}
    t_N::Int64
    mid_z::Int64
    init_state::Int64
    cr::Matrix{Float64} # covered range
    all_num_states::Int64
    len_zs::Int64


    function MarkovDefs2D3B(;dt::Float64,cov::AbstractMatrix{Float64},
        tbar::Float64,x1bar::Float64,x2bar::Float64,
        num_states::AbstractVector{Int64},mr::Float64,zs::AbstractVector{Float64},runmed_width::Int64)

        runmed_width!=0 && iseven(runmed_width) && throw(DomainError(runmed_width,"must be an odd number"))

        t_vec=collect(dt:dt:tbar)
        t_N=length(t_vec)
        len_zs=length(zs)
        mid_z=ceil(Int,len_zs/2) # ceil of num_states/2
        cr=[-x1bar x1bar;-x2bar x2bar]
        init_state=val2state([0.,0.],cr,num_states)
        all_num_states=prod(num_states)
    
        
        new(dt,cov,tbar,x1bar,x2bar,num_states,mr,zs,runmed_width,
        t_vec,t_N,mid_z,init_state,cr,all_num_states,len_zs)
    end
end

struct Markov2D3BModelParameters
    k1::Float64
    k2::Float64
    k2_intercept::Float64
    b1::Float64
    b2::Float64
    nondt::Float64
    T0sigma::Float64
    Markov2D3BModelParameters(k1::T,k2::T,k2_intercept::T,b1::T,b2::T,nondt::T,T0sigma::T) where T<:AbstractFloat = begin
        new(k1,k2,k2_intercept,b1,b2,nondt,T0sigma)
    end
    Markov2D3BModelParameters(k1::T,k2::T,k2_intercept::T,b1::T,b2::T;nondt::T=0.,T0sigma::T=0.) where T<:AbstractFloat = begin
        new(k1,k2,k2_intercept,b1,b2,nondt,T0sigma)
    end
    Markov2D3BModelParameters(x::AbstractVector{T}) where T<:AbstractFloat = begin
        if length(x)==5
            Markov2D3BModelParameters(x[1],x[2],x[3],x[4],x[5])
        elseif length(x)==6
            Markov2D3BModelParameters(x[1],x[2],x[3],x[4],x[5];nondt=x[6])
        else
            new(x...)
        end
    end
end


