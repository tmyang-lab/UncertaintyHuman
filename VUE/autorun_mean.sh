#!/bin/bash
# fixC method max_iter
# for method: NelderMead or SimulatedAnnealing 
# full rt
# julia fitall_meanrt.jl 5. NelderMead 500 12. 10. 7.
# julia fitall_meanrt.jl 1. NelderMead 500 7. 20. 14.
julia fitall_meanrt.jl 1. NelderMead 500 best
