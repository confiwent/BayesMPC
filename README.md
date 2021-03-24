# Artifact Appendix for (Paper ID: 10) Uncertainty-Aware Robust Adaptive Video Streaming with Bayesian Neural Network and Model Predictive Control

## Abstract
This document illustrates how to obtain the results shown in the paper "Uncertainty-Aware Robust Adaptive Video Streaming with Bayesian Neural Network and Model Predictive Control" which is also enclosed in the attachments. Following the instructions detailed below, the results in Figures 2, 3, 4, 5, 6 can be replicated. _Note that the badge we want to apply is the Results Replicated badge._

## The comparison algorithms
There are 5 baseline algorithms for comparison, which were detailed in Section 3.1.3 in the paper. 

These baselines correspond to 5 execution files in the "source codes'':
1. Rate-based: `rb.py`
2. Buffer-based: `bb.py`
3. BOLA: `Bola_v1.py`
4. RobustMPC: `mpc_v2.py`
5. Pensieve: `rl_no_training.py`
6. Our proposed algorithm-__BayesMPC__: `bbp_mpc_v3.py`

## The environment setup
To run the codes, some softwares and packages should be installed for replicating the results successfully. __Note that the codes of Pensieve was implemented with python2.7, and tensorflow (version = 1.11.0), and BayesMPC should run in python3.6 with pytorch (version = 1.6.0). Therefore, we suggest the reviewer to run the codes in Ubuntu and use the software "anaconda'' to manage the environments.__ For example, we can create an environment (env1) for Pensieve and another environment (env2) for BayesMPC and other baselines.

In addition, several dependent packages should be installed:
- For Pensieve \emph(env1): packages `numpy', `pandas', `matplotlib' and `tflearn' should be installed in order to run the codes. For example, you can install `numpy' by ``conda install numpy'' or ``pip install numpy'', and install `tflearn' by `pip install tflearn$==0.3.2$'. Note that it is better to install `tflearn' with the version $==0.3.2$.
- For BayesMPC and other baselines \emph(env2): packages `numpy', `pandas', `matplotlib' also should be installed. 

## Evaluation
### Results of Figure 2
To plot the results in Figure 2, you should follow the steps below:
1. create a folder named ``trace_bw_com'' in the folder ``results_lin''
2. copy the results of BayesMPC and RobustMPC in the folders ``results_lin/fcc'' and ``results_lin/HSDPA'' into the folder ``trace_bw_com''
3. plot the results by running ``python plot_results_fig2.py'' in the Terminal.