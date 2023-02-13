

"""
decision time with left/right and uncertain response
"""
# TODO:might be unnecessary to sythesis together

decisionTime_LRUR(m::ModelParameters,d::Defs)=begin
    @unpackStruct d Defs
    @unpackStruct m ModelParameters

    #_____dynamic programming________#
    
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
    l_dx=length(dx_vec)+1
    prob_dx_under_z=zeros(l_dx,len_zs)
    for i=1:len_zs
        prob_dx_under_z[:,i].=diff([0;pdf_(dx_vec,k*zs[i]*dt,sqrt(dt)*σ);1])
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
    EVnext=zero(V)
    EVnext[:,end].=-Inf

    V[:,end],D[:,end]=max_([(@view Rh_left[:,end]) (@view Rh_right[:,end]) EV_UR_vec])
    
    for iT=t_N-1:-1:1
        for (i,x) in enumerate(eachrow(xp1_state)) # expectation
            # av_x= (x .!= -1)
            # @assert all(x.!=-1)
            # EVnext[i,iT]=sum(V[x[av_x],iT+1] .* prob_dx_under_x[i,iT,av_x])
            EVnext[i,iT]=sum(V[x,iT+1] .* view(prob_dx_under_x,i,iT,:))
        end
        V[:,iT],D[:,iT]=max_([view(Rh_left,:,iT) view(Rh_right,:,iT) EV_UR_vec view(EVnext,:,iT).-c*dt])
    end

    # __________transfer D to bound______________
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


    #______________ markov fit part_________________

    g_upps=zeros(mid_z,t_N)
    g_lows=zeros(mid_z,t_N)
    g_inns=zeros(mid_z,t_N)

    num_states_i=num_states+1 # number of states with upper and lower absorbing state and inner absorbing states
    sd=incval2stateinc(σ*sqrt(dt),cr,num_states)
    ou_init_state=mid_state+1 # for there is an absorbing state
    p=zeros(num_states_i)
    cur_mat=zeros(num_states_i,num_states_i)
    for (i_z,z) in enumerate(zs[mid_z:end])
        # not related to z: initalization
        cur_mat.=diagm(ones(num_states_i))
        
        # the first step 
        inn_area=inn_upp[1]==-1 ? false : true
        
        # related to z : initialization
        kzdt=incval2stateinc(k*z*dt,cr,num_states)
        ## normal distribution
        low_b=floor(Int64,kzdt-mr*sd) # indepdent from index
        upp_b=ceil(Int64,kzdt+mr*sd) # indepdent from index

        x=collect(low_b+0.5:1:upp_b-0.5)
        y=diff([0;pdf_(x,kzdt,sd);1])
        
        for i=2:t_N

            last_low,last_upp=low[i-1],upp[i-1] # last upper/lower bound
            cur_low,cur_upp=low[i],upp[i] # current upper/lower bound

            last_inn_area=inn_area # last inner area status
            if last_inn_area
                last_inn_low,last_inn_upp=inn_low[i-1],inn_upp[i-1]
            end

            inn_area=inn_upp[i]==-1 ? false : true # current inner area status
            if inn_area
                cur_inn_low,cur_inn_upp=inn_low[i],inn_upp[i]
            end

            helpm=zeros(num_states_i,num_states_i)
            helpm[num_states_i,num_states_i]=1 # inner state
            
            for row=1:num_states # 'from' state, no absorbing state
                if row<=last_low
                    helpm[row,cur_low]=1
                    continue
                elseif row>=last_upp
                    helpm[row,cur_upp]=1
                    continue
                elseif inn_area && last_inn_area==inn_area && (row>=last_inn_low && row<=last_inn_upp)
                    helpm[row,num_states_i]=1
                    continue
                end
                index=-row+1

                for col=1:num_states # 'to' state
                    if col==cur_low # first col is special (lower absorbing boundary)
                        if row<=last_low
                            helpm[row,col]=1
                        elseif row>=last_upp # source state above (or part of) upper boundary?
                            # helpm[row,col]=0
                        elseif -row+cur_low>=low_b # entry necessary ?
                            if -row+cur_low>upp_b
                                helpm[row,col]=1
                            else
                                helpm[row,col]=sum(@view y[1:-row+cur_low-low_b+1])
                            end
                        end
                    elseif col==cur_upp # last col is special (upper absorbing boundary)
                        if row<=last_low
                            # helpm[row,col]=0
                        elseif row>=last_upp # source state above (or part of) upper boundary?
                            helpm[row,col]=1 # is absorbed by upper boundary
                        elseif -row+cur_upp<=upp_b # entry necessary ?
                            if -row+cur_upp<low_b
                                helpm[row,col]=1
                            else
                                helpm[row,col]=sum(@view y[-row+cur_upp-low_b+1:end])
                            end
                        end
                    else # standard col
                        if row<=last_low
                            # helpm[row,col]=0
                        elseif row>=last_upp
                            # helpm[row,col]=0
                        elseif col<cur_low || col>cur_upp
                            # helpm[row,col]=0
                        elseif index>=low_b && index<=upp_b
                            helpm[row,col]=y[index-low_b+1]
                        end
                    end
                    index+=1
                end # for col
            
                if inn_area
                    
                    if last_inn_area==inn_area
                        if row<last_inn_upp
                            helpm[row,num_states_i]=sum(@view helpm[row,cur_inn_upp:end-1])
                            helpm[row,cur_inn_upp:end-1].=0
                        elseif row>last_inn_low
                            helpm[row,num_states_i]=sum(@view helpm[row,1:cur_inn_low])
                            helpm[row,1:cur_inn_low].=0
                        end
                    else
                        helpm[row,num_states_i]=sum(@view helpm[row,cur_inn_low:cur_inn_upp])
                        helpm[row,cur_inn_low:cur_inn_upp].=0
                    end
                end
            end
            
            p.=cur_mat[ou_init_state,:]
            p[1:last_low].=0 # remove previously hit bounds
            p[last_upp:num_states].=0 # remove previously hit bounds
            
            if inn_area && last_inn_area==inn_area
                p[num_states_i]=0
            end

            if inn_area
                g_inns[i_z,i]=sum(p .* @view helpm[:,num_states_i])/dt
            end
            if cur_upp!=-1
                g_lows[i_z,i]=sum(p .* @view helpm[:,cur_low])/dt
                g_upps[i_z,i]=sum(p .* @view helpm[:,cur_upp])/dt
            end

            cur_mat.=cur_mat*helpm
        end
    end
    g_inns.=runmed(g_inns,runmed_width_inn)
    g_upps.=runmed(g_upps,runmed_width)
    g_lows.=runmed(g_lows,runmed_width)
    return g_upps,g_lows,g_inns
end