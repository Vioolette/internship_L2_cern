# To run this script
# > srun -p GPU -N1 --mem=8G --gres=gpu:a100:1 --ntasks-per-node=1 --time=2-0:00:00 --pty bash
# > srun -p COMPUTE -N1 --mem=16G --ntasks-per-node=1 --time=2-0:00:00 --pty bash
# > source setup.sh
# > python ML_example_analysis.py

import uproot
import os
import vector
import awkward as ak
import fastjet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, Input, Model
from tensorflow.data import Dataset, AUTOTUNE 

tf.random.set_seed(0)

data_file = '/share/data1/vfiant/processed/jet_dataset.root' # jet_dataset_minrecon_pt125_mintruth_pt200.root


def flip(x, idx):
    before = x[..., :idx]
    at = x[..., idx:idx+1]
    after = x[..., idx+1:]
    return tf.concat([before, at, after], axis=-1)


def augment_batch(X, y):

    jet_inputs = X['hits']
    jet_eta = X['eta']

    eta_flip = flip(tf.identity(jet_inputs), 2)
    phi_flip = flip(tf.identity(jet_inputs), 3)
    both_flip = flip(flip(tf.identity(jet_inputs), 2), 3)

    aug_jet_inputs = tf.concat([jet_inputs, eta_flip, phi_flip, both_flip], axis=0)
    aug_jet_eta = tf.concat([jet_eta, -jet_eta, jet_eta, -jet_eta], axis=0)

    return {'hits' : aug_jet_inputs, 'eta' : aug_jet_eta}, tf.tile(y, [4,])


with uproot.open(data_file) as f:
    matched_truth_jets = f['matched_truth_jets'].arrays()
    matched_recon_jets = f['matched_recon_jets'].arrays()
    matched_recon_jet_constituent_4vectors = f['matched_recon_jet_constituent_4vectors'].arrays()
    matched_recon_jet_constituent_3vectors = f['matched_recon_jet_constituent_3vectors'].arrays()

    matched_truth_jets = vector.zip({
        'px' : matched_truth_jets.x,
        'py' : matched_truth_jets.y,
        'pz' : matched_truth_jets.z,
        'E' : matched_truth_jets.t,
    })
    matched_recon_jets = vector.zip({
        'px' : matched_recon_jets.px,
        'py' : matched_recon_jets.py,
        'pz' : matched_recon_jets.pz,
        'E' : matched_recon_jets.E,
    })

    matched_recon_jet_constituent_4vectors = vector.zip({
        'pt' : matched_recon_jet_constituent_4vectors.rho,
        'phi' : matched_recon_jet_constituent_4vectors.phi,
        'eta' : matched_recon_jet_constituent_4vectors.eta,
        'm' : matched_recon_jet_constituent_4vectors.tau,
    })
    matched_recon_jet_constituent_3vectors = vector.zip({
        'x' : matched_recon_jet_constituent_3vectors.x,
        'y' : matched_recon_jet_constituent_3vectors.y,
        'z' : matched_recon_jet_constituent_3vectors.z,
    })
    # pour couper les pt
    mask = (matched_truth_jets.pt > 350)&(matched_recon_jets.pt > 300)
    matched_truth_jets = matched_truth_jets[mask]
    matched_recon_jets = matched_recon_jets[mask]
    matched_recon_jet_constituent_4vectors = matched_recon_jet_constituent_4vectors[mask]
    matched_recon_jet_constituent_3vectors = matched_recon_jet_constituent_3vectors[mask]
    

    


h = plt.hist2d( # 2D histogram
    ak.to_numpy(matched_truth_jets.mass), # flattening and casting to numpy is required to plot!
    ak.to_numpy(matched_recon_jets.mass),
    bins=np.logspace(1, 3, 41)
)
plt.colorbar(h[-1])
plt.plot([10, 1000], [10, 1000], color='magenta')
plt.xlabel('truth mass')
plt.ylabel('recon mass')
plt.xscale('log')
plt.yscale('log')
plt.title(f'{len(matched_recon_jets)//1000}k events')
plt.savefig('truth_vs_recon_mass.png')
plt.clf()

# scatter plot of jet constituent hits for 1st matched jet in 1st event
plt.scatter(
    matched_recon_jet_constituent_4vectors[0].deltaeta(matched_recon_jets[0]), 
    matched_recon_jet_constituent_4vectors[0].deltaphi(matched_recon_jets[0]),
    s=100 * matched_recon_jet_constituent_4vectors[0].pt,
    c=matched_recon_jet_constituent_4vectors[0].pt,
)
plt.xlabel('$\Delta\eta$')
plt.ylabel('$\Delta\phi$')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig('jet_constituents.png')
plt.clf()


def zero_pad(array, max_length=100): # max length selects the Nth highest-pT constituents of the jets
    output = ak.fill_none(ak.pad_none(array, max_length, axis=-1, clip=True), 0)
    return ak.to_numpy(output)

jet_inputs = {
    'hits' : np.stack((
                    zero_pad(matched_recon_jet_constituent_4vectors.pt), # hit pT
                    zero_pad(matched_recon_jet_constituent_3vectors.mag)/1000, # hit radius
                    zero_pad(matched_recon_jet_constituent_4vectors.deltaeta(matched_recon_jets)), # hit delta eta
                    zero_pad(matched_recon_jet_constituent_4vectors.deltaphi(matched_recon_jets)), # hit delta phi
                ), axis=-1),
    'eta' : ak.to_numpy(matched_recon_jets.eta).reshape(-1,1)
}


m_true = ak.to_numpy(matched_truth_jets.mass) # targets for NN
m_pred_baseline = ak.to_numpy(matched_recon_jets.mass) # baseline masses from FastJet

# Setup the NN
input = Input(shape=(jet_inputs['hits'].shape[1:]), name='hits')
aux_input = Input(shape=(1,), name='eta')
x = layers.Dense(32, activation='relu')(input)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Concatenate()([x, aux_input])
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

model = Model([input, aux_input], output)

model.summary()

# set the loss and gradient descent algorithm
def f_loss(y_true, y_pred):
    return tf.square(1-y_pred/y_true)
model.compile(loss=f_loss, optimizer='adam')

aug_ds = (
    Dataset.from_tensor_slices((jet_inputs, m_true))
    .batch(32)
	.map(lambda x, y: augment_batch(x, y), num_parallel_calls=AUTOTUNE)
	.prefetch(AUTOTUNE)
)


# train it will 20% of data as hold-out set

model.fit(aug_ds, epochs=20)#, validation_split=0.2) # validation_split=0.2) # % du dataset sur lequel le réseau n'est pas entrainer et sur lequel on calcul les fonctions de pertes : à changer pour voir si overfitting


# get predictions of model
model.evaluate(jet_inputs, m_true)
m_pred_NN = (model.predict(jet_inputs)).reshape(-1)


plt.hist(m_pred_baseline/m_true, bins=100, range=[0, 2], histtype='step', label=f'fastjet: stdev={np.std(m_pred_baseline/m_true):.3f}')
plt.hist(m_pred_NN/m_true, bins=100, range=[0, 2], histtype='step', label=f'NN: stdev={np.std(m_pred_NN/m_true):.3f}')
plt.legend()
plt.xlabel('pred / true mass')
plt.savefig('results.png')
plt.clf()

plt.hist(m_pred_NN, bins=150, range = [0, 300], histtype='step', label=f'NN')
plt.hist(m_true, bins=150, range = [0, 300], histtype='step', label=f'true')
plt.hist(m_pred_baseline, bins=150, range = [0, 300], histtype='step', label=f'fastjet')
plt.legend()
plt.xlabel('mass')
plt.savefig('distrib_mass.png')
plt.clf()



i = plt.hist2d( # 2D histogram
    ak.to_numpy(matched_truth_jets.mass), # flattening and casting to numpy is required to plot!
    m_pred_NN,
    bins=np.logspace(1, 3, 41)
)
plt.colorbar(i[-1])
plt.plot([10, 1000], [10, 1000], color='magenta')
plt.xlabel('truth mass')
plt.ylabel('NN mass')
plt.xscale('log')
plt.yscale('log')
plt.title(f'{len(m_pred_NN)//1000}k events')
plt.savefig('truth_vs_NN_mass.png')
plt.clf()

# The NN doesn't do any better than fastjet!
# Can you improve it?
# Currently the model uses the {pT, deta, dphi} of the first 100 jet constituents
# What else might you include in the input?
