

# one ddm model, with dropout

timeThresholdFit(mps::Markov1D2BDropoutModelParameters,d::MarkovDefs1D2BDropout)=begin
    @unpackStruct mps Markov1D2BDropoutModelParameters
    @unpackStruct d MarkovDefs1D2BDropout
    # preparation for the stable part
    upp=[b_intercept+b_slope*t for t=t_vec]
    low=[-b_intercept-b_slope*t for t=t_vec]
    dropout_rate=pdf.(truncated(Normal(drop_μ,drop_σ),drop_μ-3*drop_σ,drop_μ+3*drop_σ),t_vec)
    dropout_rate=map(enumerate(dropout_rate)) do (i,d)
        x=d/sum(dropout_rate[i:end])
        ifelse(isnan(x),0.0,x)
    end


    # the total number of states is the number of transient states plus the two absorbing states and one time absorbing state.
    g_lows=zeros(t_N,mid_z)
    g_upps=zeros(t_N,mid_z)
    g_dropouts=zeros(t_N,mid_z)
    for (i_z,z) in enumerate(zs[mid_z:end])
                    
        last_a_lower=NaN
        last_a_upper=NaN
        last_p_dropout=NaN

        kzdt=incval2stateinc(k*z*dt,cr,num_states) # We nee the increment in the state space per time step.
        sd=incval2stateinc(σ*sqrt(dt),cr,num_states) # We need the standard deviation in the state space per time step.
        low_b=floor(Int,kzdt-mr*sd)
        upp_b=ceil(Int,kzdt+mr*sd)
        y_=pdf.(Normal(kzdt,sd),low_b:upp_b)
        y_./=sum(y_)

        current_mat=Matrix{Float64}(I,num_states,num_states)

        R=zeros(num_states,3) # lower upper
        Q=zeros(num_states,num_states)
        
        for t = eachindex(t_vec)

            low_cur=lb2state(low[t],cr,num_states)
            upp_cur=ub2state(upp[t],cr,num_states)
            p_dropout_cur=dropout_rate[t]
            
            if any(low_cur.!=last_a_lower) || any(upp_cur.!=last_a_upper) || p_dropout_cur!=last_p_dropout

                last_a_lower=low_cur
                last_a_upper=upp_cur
                last_p_dropout=p_dropout_cur
            
                helpm=zeros(num_states,num_states+3)
                y=y_*(1-p_dropout_cur)

                for row=1:num_states # from state
                    index=-row
                    for col=1:num_states+3 # to state
                        if col==1 # lower bound
                            if row<=low_cur
                                helpm[row,col]=1
                            elseif row>=upp_cur
                                # helpm[row,col]=0
                            elseif -row+low_cur>=low_b
                                helpm[row,col]=ifelse(-row+low_cur>upp_b,
                                    1,sum(@view y[1:-row+low_cur-low_b+1]))
                            end
                        elseif col==num_states+2 # upper bound
                            if row<=low_cur
                                # helpm[row,col]=0
                            elseif row>=upp_cur
                                helpm[row,col]=1
                            elseif -row+upp_cur<=upp_b
                                helpm[row,col]=ifelse(-row+upp_cur<low_b,
                                    1,sum(@view y[-row+upp_cur-low_b+1:end]))
                            end
                        elseif col==num_states+3
                            helpm[row,col]=p_dropout_cur # not work?
                        else # standard col
                            if row<=low_cur
                                # helpm[row,col]=0
                            elseif row>=upp_cur
                                # helpm[row,col]=0
                            elseif col-1<=low_cur || col-1>=upp_cur
                                # helpm[row,col]=0
                            elseif index>=low_b && index<=upp_b
                                helpm[row,col]=y[index-low_b+1] 
                            end
                        end
                        index+=1
                    end # for col
                end # for row

                # construct R
                R=helpm[:,[1,num_states+2,num_states+3]]
                # construct Q
                Q=helpm[:,2:num_states+1]
            end
            current_probs=current_mat*R
            g_lows[t,i_z],g_upps[t,i_z],g_dropouts[t,i_z]=current_probs[init_state,:]./dt
            current_mat=current_mat*Q
            
        end
    end
    if runmed_width>0
        g_lows=runmed(g_lows,runmed_width)
        g_upps=runmed(g_upps,runmed_width)
    end

    return g_upps,g_lows,g_dropouts
end



# one ddm model, one race model

race2D3BFit(mps::Markov2D3BModelParameters,d::MarkovDefs2D3B)=begin
    # time invariant according to bounds
    
    @unpackStruct d MarkovDefs2D3B
    @unpackStruct mps Markov2D3BModelParameters


    # initialzation

    g1_upps=zeros(t_N,mid_z)
    g2_upps=zeros(t_N,mid_z)
    g1_lows=zeros(t_N,mid_z)

    covdt=cov_scale(cov*dt,cr,num_states) # scale covariance matrix for Markov coordinate system
    
    
    for (i_z,z) in enumerate(zs[mid_z:end])

        kzdt=incval2stateinc([k1*z,k2*(k2_intercept-z)].*dt,cr,num_states)
        
        # we need the initial values in the state space.

        t_vec=collect(dt:dt:tbar)

        low_b1=floor(Int,kzdt[1]-mr*sqrt(covdt[1,1]))
        upp_b1=ceil(Int,kzdt[1]+mr*sqrt(covdt[1,1]))
        low_b2=floor(Int,kzdt[2]-mr*sqrt(covdt[2,2]))
        upp_b2=ceil(Int,kzdt[2]+mr*sqrt(covdt[2,2]))

        x1=low_b1:upp_b1
        x2=low_b2:upp_b2
        y_=map(collect.(Iterators.product(
            low_b1-0.25:0.5:upp_b1+0.25,
            low_b2-0.25:0.5:upp_b2+0.25))) do x
                pdf(MvNormal(kzdt,covdt),x)
            end #evaluate the PDF
        y_=y_'./sum(y_) # normalize the transition probabilities
        y=zeros(length(x2),length(x1))


        for i in eachindex(x2), j in eachindex(x1)
            y[i,j]=sum(@view y_[(i-1)*2+1:i*2,(j-1)*2+1:j*2]) # 4 points in the calculated PDF contribute to each value in the transition "block".
        end

        # ______________because the boundary is invariant____________
        cur_bs=zeros(all_num_states,3)
        for i=1:all_num_states
            val=state2val(i,cr,num_states)
            cur_bs[i,:]=[(val.>[b1,b2])...,val[1]<-b1]
        end
        
        ## transition matrix is only created once.
        Q=zeros(all_num_states,all_num_states) # this will be the transition matrix.
        R=zeros(all_num_states,3)
        current_probs=CUDA.zeros(all_num_states,3)
        current_mat=cu(Matrix{Float32}(I,all_num_states,all_num_states)) # start with a sparse identity matrix

        for from_state=1:all_num_states # This is the "from" state
            from_row,from_col=state2rowcol(from_state,num_states) # get the location of the "from" state on the grid of transient states

            for row_ind in eachindex(x2), col_ind in eachindex(x1)
                to_row=from_row+x2[row_ind] # This is the row of the "to" state
                to_col=from_col+x1[col_ind] # This is the column of the "to" state
                to_state=rowcol2state(to_row,to_col,num_states) # This is the "to" state

                if to_state==0 # out of range
                    val=rowcol2val(to_row,to_col,cr,num_states)
                    b_crossed=[(val.>[b1,b2])...,val[1]<-b1]
                    num_b_crossed=sum(b_crossed)
                    if num_b_crossed==0
                        temp_state=nearest_state(to_row,to_col,num_states)
                        Q[from_state,temp_state]+=y[row_ind,col_ind]
                    else
                        R[from_state,:].+=ifelse.(b_crossed,y[row_ind,col_ind]/num_b_crossed,0)
                    end
                else # within range
                    num_a_cur=sum(@view cur_bs[to_state,:])
                    if num_a_cur==0
                        Q[from_state,to_state]+=y[row_ind,col_ind]
                    else
                        R[from_state,:].+=ifelse.(cur_bs[to_state,:].!=0,y[row_ind,col_ind]/num_a_cur,0)                            
                    end
                end
            end # for row & col ind
        end # for all "from" states
        # _____________end_because the boundary is invariant____________

        # for each time point
        Q=cu(Q)
        R=cu(R)
        for t in eachindex(t_vec)
            @CUDA.sync current_probs=current_mat*R
            @CUDA.allowscalar g1_upps[t,i_z],g2_upps[t,i_z],g1_lows[t,i_z]=(@view current_probs[init_state,:])/dt
            @CUDA.sync current_mat*=Q
        end # loop

    end
    if runmed_width>0
        g1_upps=runmed(g1_upps,runmed_width)
        g2_upps=runmed(g2_upps,runmed_width)
        g1_lows=runmed(g1_lows,runmed_width)
    end
    g1_upps,g1_lows,g2_upps
end