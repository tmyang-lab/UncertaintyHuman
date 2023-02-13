"""
the decision area is given
reaction time fitting using Markov method 
"""
markovFit(bps::BoundPositions,mps::ModelParameters,d::Defs)=begin
    @unpackStruct d Defs
    @unpackStruct mps ModelParameters
    @unpackStruct bps BoundPositions

    g_upps=zeros(t_N,mid_z)
    g_lows=zeros(t_N,mid_z)
    g_inns=zeros(t_N,mid_z)

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

        x=collect(low_b:1.0:upp_b)
        y=pdf_(x,kzdt,sd)
        
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
                g_inns[i,i_z]=sum(p .* @view helpm[:,num_states_i])/dt
            end
            if cur_upp!=-1
                g_lows[i,i_z]=sum(p .* @view helpm[:,cur_low])/dt
                g_upps[i,i_z]=sum(p .* @view helpm[:,cur_upp])/dt
            end

            cur_mat.=cur_mat*helpm
        end
    end
    g_inns.=runmed(g_inns,runmed_width_inn)
    g_upps.=runmed(g_upps,runmed_width)
    g_lows.=runmed(g_lows,runmed_width)
    return g_upps,g_lows,g_inns
end