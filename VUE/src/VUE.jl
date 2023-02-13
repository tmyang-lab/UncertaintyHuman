module VUE
# using Distributed,PaddedViews
using Parameters,LazyGrids,Distributions
using SpecialFunctions,LinearAlgebra,Statistics
# LoopVectorization
using UnPack
using NaNStatistics
using SparseArrays
using CUDA

include("Param.jl")
include("DP.jl")
include("GenerateRT.jl")
include("DataProcessing.jl")
include("SimulationFitted.jl")

# fitting 
export 
    findnearest,
    decisionArea,decisionBound,decisionAreaFixed,markovFit,
    runmed,incval2stateinc,state2val,
    # Params.jl
    Defs,ModelParameters,BoundPositions,defs,defs2, # constant
    MarkovDefs1D2BDropout,Markov1D2BDropoutModelParameters, # time threshold model
    MarkovDefs2D3B,Markov2D3BModelParameters, # 2D3B model
    # MarkovFitOthers.jl
    timeThresholdFit,  # time threshold model
    race2D3BFit,   # 2D3B model
    decisionTime_LRUR,pdf_,max_,erf,# for test
    # DataProcessing.jl
    subjMean
end