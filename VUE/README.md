

# Value-based Uncertainty Estimation (VUE) model

To fit the model, you should activate and precompile the `VUE` package in julia

## Runfiles

### data file

`gen_data_for_fitting.jl`

`gen_simu.py`

### fitting files

`autorun_mean.sh`

`autorun_mean_stdR.sh` : it should be run after `autorun_mean.sh`

`autorun_mean_tt.sh`

`autorun_mean_2D3B.sh`

### covertion files

covnert julia generated results to python readable files for plotting

`jl-connect.py`

`jl-connect-others.py`

`replication-on-kiani09.ipynb (julia)`

`gen_others_fig_data.jl`

If you have any problem run these files, please contact flumer@qq.com