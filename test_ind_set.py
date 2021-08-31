from tqdm import tqdm
import mldp_seqboot
import ind_set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import timeit

def mc_test(numObs, numBars, maxH):
    tx = mldp_seqboot.getRndT1(numObs, numBars, maxH)
    ix = ind_set.Indicator(tx)

    std_boot = ix.standard_bootstrap()
    seq_boot = ix.sequential_bootstrap()

    std_u = ix.sample_avg_uniqueness(std_boot)
    seq_u = ix.sample_avg_uniqueness(seq_boot)

    return { 'stdU': std_u, 'seqU': seq_u } 


def exp_trials(num_trials=100000):
    
    out = []
    for i in tqdm(range(num_trials)):
        mx = mc_test(numObs=10, numBars=100, maxH=5)
        out.append(mx)
    
    out = pd.DataFrame(out)
    px = out.plot.kde()
    px.set_xlim(0, 1)
    plt.savefig('exp_trials.png')
    print('MEDIANS:')
    print(out.median())
    print('ANOVA:')
    print(f_oneway(out['seqU'], out['stdU']))
    return out


def tx_seqboot(tx):
    ix = ind_set.Indicator(tx)
    ix.sequential_bootstrap()
    return None

def tx_mldp_seqboot(t1):
    barIx=range(t1.max()+1)
    indM=mldp_seqboot.getIndMatrix(barIx,t1)
    mldp_seqboot.seqBootstrap(indM)    
    return None

def time_trials(configs=[(10, 100, 5), (20, 200, 10), (30, 300, 15), (40, 400, 20), (50, 500, 25), (60, 600, 30), (70, 700, 35), (80, 800, 40), (90, 900, 45), (100, 1000, 50), (200, 2000, 100), (300, 3000, 150)], num_runs=10):

    out = []
    pbar = tqdm(total=len(configs)*2)
    for (numObs, numBars, maxH) in configs:

        print((numObs, numBars, maxH))
        tx = mldp_seqboot.getRndT1(numObs, numBars, maxH)
        
        time_seqboot = timeit.Timer(lambda: tx_seqboot(tx)).timeit(num_runs) / num_runs
        pbar.update(1)
        time_mldp_seqboot = timeit.Timer(lambda: tx_mldp_seqboot(tx)).timeit(num_runs) / num_runs
        pbar.update(1)
        out.append({'numObs': numObs, 'numBars': numBars, 'maxH': maxH,  'seq': time_seqboot, 'mldp': time_mldp_seqboot })
        pbar.refresh()


    pbar.close()
    return pd.DataFrame(out)

#exp_trials()
#print(time_trials())
