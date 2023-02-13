'''
Filename: /home/flumer/Documents/Project/to_lxd_7subj_20201025/settings.py
Path: /home/flumer/Documents/Project/to_lxd_7subj_20201025
Created Date: Monday, October 26th 2020, 4:56:52 pm
Author: LI Xiaodong

Copyright (c) 2020 Your Company
'''

import matplotlib as mpl


analysis_cols=['choosedDirection',
       'correct', 'trialNo','trialType', 'dots.startTime', 'dots.coherence',
       'dots.finalcoherence', 'dots.direction',
       'dots.dircoherence',  'dots.dirfinalcoherence','_dots.me',
       'firstStageRt', 'maxWaitTimeAfterRel',
       'timeBtwnDnRlsAndRsp', 'secondStageRt', 'secondChoiceTimeAfterUpRls',
       'totalRt', 'errorStatus', 'intoStage2', 'delayTime','participant','session', 'date', 'expName', 'psychopyVersion', 'frameRate']
model_cols=['choosedDirection',
        'correct', 'trialNo','trialType', 'dots.coherence','_dots.me',
        'dots.direction','dots.dircoherence',
        'firstStageRt','secondStageRt', 
        'totalRt', 'intoStage2', 'delayTime']
# conditions map

"""
first term:
    no-up/opt-up/delay[time]
second term:
    lr/up/2nd
"""
color_map={
    'no-up lr':'green',
    'no-up quick':'lawngreen',
    'no-up slow':'darkolivegreen',
    'no-up lr correct':'green',
    'no-up lr error':'red',
    'no-up':'green',
    'opt-up':'black',
    'opt-up up':'orange',
    'opt-up lr':'blue',
    'opt-up 2nd':'purple',
    
    'delay0':'black',
    'delay250':'darkblue',
    'delay500':'blue',
    'delay1000':'cyan',

    'delay0 lr':'black',
    'delay250 lr':'darkblue',
    'delay500 lr':'blue',
    'delay1000 lr':'cyan',
    'delay0 up':'darkred',
    'delay250 up':'chocolate',
    'delay500 up':'darkorange',
    'delay1000 up':'orange'
}

label_map={
    
    'no-up':'PD in STD',
    'no-up lr':'PD in STD',
    'no-up quick':'PD in STD quick',
    'no-up slow':'PD in STD slow',
    'no-up lr correct':'PD in STD correct',
    'no-up lr error':'PD in STD error',

    'opt-up':'CUE',
    'opt-up up':'UR in CUE',
    'opt-up 2nd':'PD2 in CUE',
    'opt-up lr':'PD in CUE',

    'delay0':'delay0',
    'delay250':'delay250',
    'delay500':'delay500',
    'delay1000':'delay1000',
    'delay0 lr':'delay0 left/right',
    'delay250 lr':'delay250 left/right',
    'delay500 lr':'delay500 left/right',
    'delay1000 lr':'delay1000 left/right',
    'delay0 up':'delay0 UR',
    'delay250 up':'delay250 UR',
    'delay500 up':'delay500 UR',
    'delay1000 up':'delay1000 UR',

    'up':'UR',
    'correct':'correct',
    'wrong':'wrong'
}

color_degree_map={
    'no-up lr':'Greens',
    'opt-up':'Greys',
    'opt-up up':'Oranges',
    'opt-up lr':'Blues',
    'opt-up 2nd':'Purples',

    'delay0 lr':'Greys',
    'delay250 lr':'PuBu',
    'delay500 lr':'Blues',
    'delay1000 lr':'GnBu'
}

def get_color(coh,d):
    """
    different color on coherence
    """
    cmap=mpl.cm.get_cmap(color_degree_map[d])
    return cmap((0.01+coh)/0.26*100)