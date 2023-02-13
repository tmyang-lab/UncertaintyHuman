#!/bin/bash
# k,b_intercept,b_slope,drop_μ,drop_σ
# for method: NelderMead or SimulatedAnnealing 
# full rt

# julia fitall_meanrt_tt.jl 1 NelderMead 500 12 1.0 0.1 1.0 0.01
julia fitall_meanrt_tt.jl 1. NelderMead 500 best
