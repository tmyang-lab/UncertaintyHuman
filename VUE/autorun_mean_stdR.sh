#!/bin/bash
# fixC method max_iter
# for method: NelderMead or SimulatedAnnealing 
# full rt
#julia fitall_meanrt_stdR.jl 5. SimulatedAnnealing 100 12.
#julia fitall_meanrt_stdR.jl 5. SimulatedAnnealing 100 18.
#julia fitall_meanrt_stdR.jl 5. SimulatedAnnealing 100 6.
#julia fitall_meanrt_stdR.jl 5. SimulatedAnnealing 100 15.
#julia fitall_meanrt_stdR.jl 5. SimulatedAnnealing 100 9.
julia fitall_meanrt_stdR.jl 1. SimulatedAnnealing 200 best
