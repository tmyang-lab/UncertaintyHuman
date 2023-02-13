

### markov fit
# val2state(val::Float64,cr::Vector{Float64},num_states::Int64)=begin
#     temp=(val .- cr[1] .+ cr[2] .- cr[1])/(2*(cr[2]-cr[1]))
#     if temp<1.5 temp=1 end
#     if temp>num_states-0.5 temp=num_states end
#     state=round(Int64,temp)
#     return state
# end

state2val(state::Int,cr::Vector{Float64},num_states::Int64)=begin
    return cr[1]+(state-0.5)*(cr[2]-cr[1])/num_states
end

@inline incval2stateinc(inc_val::Float64,cr::Vector{Float64},num_states::Int64)=begin
    return inc_val/(cr[2]-cr[1])*num_states
end


runmed(unfiltered::AbstractVector{Float64},filter_width::Int64)=begin
    n=length(unfiltered)
    filtered=zeros(n)

    start_full=(filter_width+1)÷2
    half_width=(filter_width-1)÷2

    for i=start_full:n-start_full
        filtered[i]=median(@view unfiltered[i-half_width:i+half_width])
    end
    return filtered
end

runmed(unfiltered::AbstractMatrix{Float64},filter_width::Int64)=begin
    n,m=size(unfiltered)
    
    filtered=zeros(n,m)

    start_full=(filter_width+1)÷2
    half_width=(filter_width-1)÷2
    for i=1:m 
        for j=start_full:n-start_full
            filtered[j,i]=median(@view unfiltered[j-half_width:j+half_width,i])
        end
    end
    return filtered
end


# MarkovFitOthers
# dropout
lb2state(val::Float64,cr::AbstractVector{Float64},num_states::Int)=begin
    temp=(val-cr[1])/(cr[2]-cr[1])*(num_states)
    temp<.5 && (temp=0)
    state=round(Int,temp)
    return state
end

ub2state(val::Float64,cr::AbstractVector{Float64},num_states::Int)=begin
    temp=(val-cr[1])/(cr[2]-cr[1])*(num_states)+1;
    temp>num_states+0.5 && (temp=num_states+1)
    state=round(Int,temp)
    return state
end

# 2d_3b
incval2stateinc(inc_val::AbstractVector{Float64},cr::AbstractMatrix{Float64},num_states::AbstractVector{Int})=begin
    delta_col=inc_val[1]/(cr[1,2]-cr[1,1])*num_states[1]
    delta_row=inc_val[2]/(cr[2,2]-cr[2,1])*num_states[2]
    state_inc=[delta_col,delta_row]
    return state_inc
end

state2val(state::Int,cr::AbstractMatrix{Float64},num_states::AbstractVector{Int})=begin
    row=floor((state-1)/num_states[1])+1
    col=state-(row-1)*num_states[1]
    val1=cr[1,1]+(col-0.5)*(cr[1,2]-cr[1,1])/num_states[1]
    val2=cr[2,1]+(row-0.5)*(cr[2,2]-cr[2,1])/num_states[2]
    return val1,val2
end

cov_scale(covm::AbstractMatrix{Float64},cr::AbstractMatrix{Float64},num_states::AbstractVector{Int})=begin
    new_cov=zeros(2,2)
    div1=(cr[1,2]-cr[1,1])/num_states[1]
    div2=(cr[2,2]-cr[2,1])/num_states[2]
    new_cov[1,1]=covm[1,1]/div2^2
    new_cov[1,2]=covm[1,2]/(div1*div2)
    new_cov[2,1]=covm[2,1]/(div1*div2)
    new_cov[2,2]=covm[2,2]/div2^2
    return new_cov
end

rowcol2state(row::Int,col::Int,num_states::AbstractVector{Int})=begin
    # state=0 if out of range
    state=col<1 || col>num_states[1] || row<1 || row>num_states[2] ? 0 : (row-1)*num_states[1]+col
    return state
end

state2rowcol(state::Int,num_states::AbstractVector{Int})=begin
    row=floor(Int,(state-1)/num_states[1])+1
    col=state-(row-1)*num_states[1]
    return row,col
end

rowcol2val(row::Int,col::Int,cr,num_states::AbstractVector{Int})=begin
    val1=cr[1,1]+(col-0.5)*(cr[1,2]-cr[1,1])/num_states[1]
    val2=cr[2,1]+(row-0.5)*(cr[2,2]-cr[2,1])/num_states[2]
    return [val1,val2]
end

nearest_state(row::Int,col::Int,num_states::AbstractVector{Int})=begin
    # get the nearest transient state for an out-of-range location
    if row>num_states[2] && col<1 # upper left quadrant
        state=rowcol2state(num_states[2],1,num_states)
    elseif row>num_states[2] && col>=1 && col<=num_states[1] # upper part
        state=rowcol2state(num_states[2],col,num_states)
    elseif row>num_states[2] && col>num_states[1] # upper right quadrant
        state=rowcol2state(num_states[2],num_states[1],num_states)
    elseif row>=1 && row<=num_states[2] && col>num_states[1] # right part
        state=rowcol2state(row,num_states[1],num_states)
    elseif row<1 && col>num_states[1] # lower right quadrant
        state=num_states[1]
    elseif row<1 && col>=1 && col<=num_states[1] # lower part
        state=col
    elseif row<1 && col<1 # lower left quadrant
        state=1
    else # left part
        state=rowcol2state(row,1,num_states)
    end
end