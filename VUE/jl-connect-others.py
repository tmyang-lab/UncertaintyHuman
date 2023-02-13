

import numpy as np

subjs=["s01","s02","s03","s06","s08","s09","s10","s12","s13","s15"]
for main_folder in ["../connect/data_tt/","../connect/data_2d3b/"]:
    for subj in subjs:
        
        npz_rt=dict(np.load(f"{main_folder}{subj}_rt.npz"))
        npz_rt["xlabel"]="Coherence",
        npz_rt["ylabel"]="Reaction time (s)"

        npz_ur_prop=dict(np.load(f"{main_folder}{subj}_ur_prop.npz"))
        npz_ur_prop["xlabel"]="|Coherence|",
        npz_ur_prop["ylabel"]="UR proportion"

        print(f"subject {subj}")
        d={"rt":npz_rt,"ur_prop":npz_ur_prop}
        np.save(f"{main_folder}{subj}.npy",d)
