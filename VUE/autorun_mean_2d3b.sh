#!/bin/bash
# k1,k2,k2_intercept,b1,b2,nondt
# for method: NelderMead or SimulatedAnnealing 
# mean rt

#julia fitall_meanrt_2D3B.jl 1 NelderMead 500 12 5.0 1.0 1.0 1.0
julia fitall_meanrt_2D3B.jl 1. NelderMead 500 best
