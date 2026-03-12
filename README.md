trainingRNNs (PyTorch port)
==========================

This is a small PyTorch reimplementation of Razvan Pascanu's 2013 Theano code:
https://github.com/pascanur/trainingRNNs

It includes:
  - Vanilla (Elman) RNN model with the same heads:
      * lastSoftmax, softmax (all steps), lastLinear
  - The same synthetic tasks:
      * torder, torder3, add, mul, mem, perm
  - The same training loop structure:
      * variable length sampled between min_length and max_length per iteration
      * evaluation every checkFreq iterations with ebs examples in chunks of cbs
  - "Rescale" gradient clipping (global norm) like the original code
  - The paper's gradient-norm regularizer Omega applied to W_hh only (alpha > 0)

Requirements
------------
  pip install torch numpy

Quick start
-----------
Train memorization task (defaults similar to RNN.py main):
  python train.py --task mem --init smart_tanh --nhid 50 --min_length 50 --max_length 200 \
      --bs 20 --ebs 10000 --cbs 1000 --alpha 2.0 --lr 0.01 --clipstyle rescale --cutoff 1.0

Other tasks:
  python train.py --task torder
  python train.py --task add
  python train.py --task perm

Outputs
-------
At the end it saves <name>_final_state.npz with the same arrays as the Theano code:
  train_nll, valid_error, gradient_norm, rho_Whh, Omega plus the weights.

Notes
-----
- Minor numerical differences vs Theano are expected (different BLAS, random streams, float behavior).
- The regularizer Omega matches the original computation path and is applied only
  to W_hh (like the Theano code that adds alpha * dOmega/dW_hh to gW_hh).
