# To run this script
# > srun -p GPU -N1 --mem=256G --gres=gpu:a100:1 --ntasks-per-node=1 --time=2-0:00:00 --pty bash
# > source setup.sh
# > python example_analysis.py

import uproot
import os
import vector
from halo import Halo
import awkward as ak
import fastjet
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/share/data1/vfiant/processed/'

@Halo(text='Reading cells', spinner='line')
def read_cells(dir, filter_by='.root'):
    cells = uproot.concatenate(
        [data_dir+f+':cells' for f in os.listdir(dir) if filter_by in f],
        ['cell_et', 'cell_x', 'cell_y', 'cell_z', 'cell_sampling']
    )
    cell_position = vector.zip({
        'x' : cells.cell_x,
        'y' : cells.cell_y,
        'z' : cells.cell_z,
    })

    cell_4vectors = vector.zip({
        'eta' : cell_position.eta,
        'phi' : cell_position.phi,
        'pt' : cells.cell_et / 1000,
        'm' : ak.zeros_like(cells.cell_et)
    })
    return cell_4vectors

def read_truth(dir, filter_by='.root'):
    truth = uproot.concatenate(
        [data_dir+f+':truth' for f in os.listdir(dir) if filter_by in f],
    )
    truth_jets = vector.zip({
        'E' : truth.truth_jet_antikt_10_E,
        'px' : truth.truth_jet_antikt_10_px,
        'py' : truth.truth_jet_antikt_10_py,
        'pz' : truth.truth_jet_antikt_10_pz, 
    })
    return truth_jets

def truth_match(recon_jets, truth_jets, max_dR=0.4):
    left, right = ak.unzip(ak.argcartesian([recon_jets, truth_jets]))

    dR = recon_jets[left].deltaR(truth_jets[right])
    is_close = dR < max_dR

    matched_truth_jets = truth_jets[right][is_close]
    matched_recon_jets = recon_jets[left][is_close]

    return matched_recon_jets, matched_truth_jets

def efficiency(truth_jets, matched_truth_jets):
    right = ak.unzip(ak.argcartesian([truth_jets]))
    eff = []
    for i in range(len(truth_jets)):
        if len(truth_jets[right][i]) != 0:
            eff.append(len(matched_truth_jets[i])/len(truth_jets[i]))
        else:
            eff.append(0)
    return(eff)

def purity(recon_jets, matched_recon_jets):
    left = ak.unzip(ak.argcartesian([recon_jets]))
    pur = []
    for i in range(len(recon_jets)):
        if len(recon_jets[left][i]) != 0:
            pur.append(len(matched_recon_jets[i])/len(recon_jets[i]))
        else:
            pur.append(0)
    return(pur)

def mass_bulk_distribution(matched_truth_jets, matched_recon_jets):
    b = []
    for i in range(len(matched_truth_jets)):
        rap = (matched_recon_jets[i].mass) / (matched_truth_jets[i].mass)
        for el in rap:
            b.append(el)
    return(b)

def min_dR(truth_jets, recon_jets):
    min_dR = []
    pt_true_dR = []
    pt_recon_dR = []
    left, right = ak.unzip(ak.argcartesian([recon_jets, truth_jets]))
    for i in range(len(truth_jets[right])):
        collision = truth_jets[right][i]
        for jet_t in collision:
            r_min = 100
            pt_t = jet_t.pt
            pt_r = 1
            test = False
            for jet_r in recon_jets[left][i]:
                r = jet_r.deltaR(jet_t)
                if r<r_min:
                    test = True
                    r_min = r
                    pt_r = jet_r.pt
            if test:
                min_dR.append(r_min)
                pt_true_dR.append(pt_t)
                pt_recon_dR.append(pt_r)
    return(min_dR, pt_true_dR, pt_recon_dR)

cell_4vectors = read_cells(data_dir, filter_by='k.root') #, filter_by='_5k.root') # running over only one file for speed
truth_jets = read_truth(data_dir, filter_by='k.root') # , filter_by='_5k.root') # replace this argument if you want to run over all 82k

jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1)
cluster = fastjet.ClusterSequence(cell_4vectors, jetdef)
recon_jets = cluster.inclusive_jets(min_pt=50)

matched_recon_jets, matched_truth_jets = truth_match(recon_jets, truth_jets)

h = plt.hist2d( # 2D histogram
    ak.to_numpy(ak.flatten(matched_truth_jets.mass)), # flattening and casting to numpy is required to plot!
    ak.to_numpy(ak.flatten(matched_recon_jets.mass)),
    bins=np.logspace(1, 3, 41)
)
plt.colorbar(h[-1])
plt.xlabel('truth mass')
plt.ylabel('recon mass')
plt.xscale('log')
plt.yscale('log')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('truth_vs_recon_mass.png')
plt.clf()

m =  plt.hist(ak.to_numpy(ak.flatten(matched_recon_jets.mass)), bins=50)
plt.xlim(0, 250)
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('recon_mass.png')
plt.clf()

e = plt.hist(efficiency(truth_jets, matched_truth_jets), bins=50)
plt.xlabel('efficiency = number matched jets / number truth jets')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('efficiency.png')
plt.clf()

p = plt.hist(purity(recon_jets, matched_recon_jets), bins=50)
plt.xlabel('purity = number matched jets / number recon jets')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('purity.png')
plt.clf()

b = plt.hist(mass_bulk_distribution(matched_truth_jets, matched_recon_jets), bins=50)
plt.xlabel('mass recon / mass truth')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('rap_mass.png')
plt.clf()


res = min_dR(truth_jets, recon_jets)
dr_pt_t = plt.plot(res[1], res[0], 'o', linestyle='None', markeredgewidth=0.5, alpha = 1)
plt.xlabel('pt truth')
plt.ylabel('dR minimum')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('min_dR_vs_pt_truth.png')
plt.clf()

dr_pt_r = plt.plot(res[2], res[0], 'o', linestyle='None', markeredgewidth=0.5, alpha = 1)
plt.xlabel('pt recon')
plt.ylabel('dR minimum')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('min_dR_vs_pt_recon.png')
plt.clf()

dr = plt.hist(res[0], bins=50)
plt.xlim(0, 3)
m = np.mean(res[0])
plt.axvline(m, color = 'red')
min_ylim, max_ylim = plt.ylim()
plt.text(m*1.02, max_ylim*0.9, 'Mean: {:.2f}'.format(m))
plt.xlabel('dR')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('hist_dr.png')
plt.clf()
