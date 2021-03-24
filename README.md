# Artifact Appendix for (Paper ID: 10) Uncertainty-Aware Robust Adaptive Video Streaming with Bayesian Neural Network and Model Predictive Control

## Abstract
This document illustrates how to obtain the results shown in the paper "Uncertainty-Aware Robust Adaptive Video Streaming with Bayesian Neural Network and Model Predictive Control" which is also enclosed in the attachments. Following the instructions detailed below, the results in Figures 2, 3, 4, 5, 6 can be replicated. _Note that the badge we want to apply is the Results Replicated badge._

## The comparison algorithms
There are 5 baseline algorithms for comparison, which were detailed in Section 3.1.3 in the paper. 

These baseline algorithms correspond to 5 execution files in the "source codes'':
1. Rate-based: `rb.py`
2. Buffer-based: `bb.py`
3. BOLA: `Bola_v1.py`
4. RobustMPC: `mpc_v2.py`
5. Pensieve: `rl_no_training.py`
6. Our proposed algorithm-_BayesMPC_: `bbp_mpc_v3.py`

## The environment setup
To run the codes, some softwares and packages should be installed for replicating the results successfully. In addition, it is suggested to run the codes with Ubuntu 16.04/18.04.

### Intall anaconda to manage the test environments.
- Download the anaconda installers from the [official website](https://www.anaconda.com/products/individual#Downloads), generally choose the 64-Bit (x86) Installer.
- Open Terminal, install anaconda

    ```
    cd Downloads
    bash Anaconda3-2020.11-Linux-x86_64.sh
    ```
- Type `yes` when there are questions in the installation procedure.
- Update `.bashrc`

    ```
    source ~/.bashrc
    ```

- Type `python` in Terminal, the installation is successful if the Anaconda logo shows in the terminal.

### Intall prerequisities for _BayesMPC_
The _BayesMPC_ should be tested with python 3.6, pytorch 1.6.0, numpy, matplotlib, and pandas.
- Create a new virtual environment named _bayes_ for testing _BayesMPC_

    ```
    conda create --name bayes python=3.6
    ```
- Activate the virtual environment and intall the packages in it.

    ```
    conda activate bayes
    conda install -n bayes numpy pandas matplotlib
    ```
- Install PyTorch. Note that the command of PyTorch intallation depends on the actual compute platform of your own computer, and you can choose appropriate version following the [guide page](https://pytorch.org/get-started/locally/). For example, if you have intalled `CUDA 10.2`, you can intall PyTorch with the latest version by running this Command:

    ```
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```

- You can deactivate the current virtual environment by running

    ```
    conda deactivate
    ```

### Intall prerequisities for Pensieve
The Pensieve should be tested with python 2.7, Tensorflow (version <= 1.11.0), TFLearn (version <= 0.3.2), numpy, matplotlib, and pandas.
- Create a new virtual environment named _pensieve_ for testing Pensieve

    ```
    conda create --name pensieve python=2.7
    ```
- Activate the virtual environment and intall the packages in it.

    ```
    conda activate pensieve
    conda install -n bayes numpy pandas matplotlib
    ```
- Install Tensorflow.

    ```
    conda install tensorflow==1.11.0
    ```
- Install TFLearn.

    ```
    pip install tflearn==0.3.2
    ```
- You can deactivate the current virtual environment by running

    ```
    conda deactivate
    ```

### Intall prerequisities for other baseline algorithms
The other baseline algorithms can be tested in _pensieve_.

## Evaluation
### Results of Fig.2
To replicate the results in Fig.2, you should follow the steps below:
1. Create two results folders

    ```
    mkdir ./results_lin
    mkdir ./results_lin/cb_fcc ./results_lin/cb_HSDPA
    ```
2. Generate the results of _BayesMPC_ and RobustMPC in Fig.2(a) and (b).

    ```python
    conda activate bayes
    ## for fig.2(a)
    python bbp_mpc_v3.py --cb --HSDPA
    python mpc_v2.py --cb --HSDPA
    ## for fig.2(b)
    python bbp_mpc_v3.py --cb --FCC
    python mpc_v2.py --cb --FCC
    ```
3. Plot the results in Fig.2(a).

    ```python
    ## for fig.2(a)
    python plot_results_fig2.py --a
    ## for fig.2(b)
    python plot_results_fig2.py --b
    ```

<!-- ![fig2a](./pic/random_traces_prediction_norway.pdf)
![fig2b](./pic/random_traces_prediction_fcc.pdf) -->

### Results of Figs. 3, 4
There are two different QoE metrics: $QoE_{lin}$ and $QoE_{log}$ and two different throughput datasets: FCC and HSDPA. For comparison, all $6$ baseline algorithms should be tested in these settings and the results shown in Figs.3, 4 in Section.3 of the paper can be replicated by following commands:
1. Create the results folders for different QoE metrics and different throughput datasets.

    ```python
    ## create folder for QoE metric QoE_lin
    mkdir ./results_lin
    ## create folder for different datasets with the QoE_lin metric
    mkdir ./results_lin/fcc ./results_lin/HSDPA
    ## same for QoE metric QoE_log
    mkdir ./results_log
    mkdir ./results_log/fcc ./results_log/HSDPA
    ```

2. Test _BayesMPC_ for different QoE metrics and for different throughput datasets.

    ```python
    conda activate bayes
    python bbp_mpc_v3.py --lin --HSDPA
    python bbp_mpc_v3.py --lin --FCC
    python bbp_mpc_v3.py --log --HSDPA
    python bbp_mpc_v3.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

3. Test Pensieve for different QoE metrics and for different throughput datasets.

    ```python
    conda activate pensieve
    python rl_no_training.py --lin --HSDPA
    python rl_no_training.py --lin --FCC
    python rl_no_training.py --log --HSDPA
    python rl_no_training.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

4. Test other baseline algorithms in ```pensieve``` virtual environment. Note that the following commands are illustrated for testing RobustMPC, and other baseline algorithms can be tested by just replacing the file name ```mpc_v2.py``` with the corresponding file name (```Bola_v1.py```, ```bb.py```, ```rb.py```).

    ```python
    conda activate pensieve
    python mpc_v2.py --lin --HSDPA
    python mpc_v2.py --lin --FCC
    python mpc_v2.py --log --HSDPA
    python mpc_v2.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

5. __Plot the results__: The results in Figs. 3 and 4 for different QoE metrics and different datasets can be plotted by the following commands:

    ```python
    conda activate pensieve
    # for results tested with QoE_lin and HSDPA dataset in Fig. 3 and 4(a)
    python plot_results_fig34.py --lin --HSDPA
    # for results tested with QoE_log and HSDPA dataset in Fig. 3 and 4(b)
    python plot_results_fig34.py --log --HSDPA
    # for results tested with QoE_lin and FCC dataset in Fig. 3 and 4(c)
    python plot_results_fig34.py --lin --FCC
    # for results tested with QoE_log and FCC dataset in Fig. 3 and 4(d)
    python plot_results_fig34.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

### Results of Fig.6
The procedure for plotting Figure 6 is similar to the procedure for Figure 3, 4. The only difference is test throughput dataset for baseline algorithms. The results of Fig.6 are tested with Oboe dataset.

- Test _BayesMPC_ with Oboe dataset.

    ```python
    conda activate bayes
    python bbp_mpc_v3.py --lin --Oboe
    python bbp_mpc_v3.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```

- Test Pensieve with Oboe dataset. 

    ```python
    conda activate pensieve
    python rl_no_training.py --lin --Oboe
    python rl_no_training.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```

- Test other algorithms with Oboe dataset. Note that the following commands are illustrated for testing RobustMPC, and other baseline algorithms can be tested by just replacing the file name ```mpc_v2.py``` with the corresponding file name (```Bola_v1.py```, ```bb.py```, ```rb.py```).

    ```python
    conda activate pensieve
    python mpc_v2.py --lin --Oboe
    python mpc_v2.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```

- __Plot the results__: The results in Fig.6 for different QoE metrics with Oboe datasets can be plotted by the following commands:

    ```python
    conda activate pensieve
    # for results tested with QoE_lin and Oboe dataset in Fig. 6(a)
    python plot_results_fig34.py --lin --Oboe
    # for results tested with QoE_log and Oboe dataset in Fig. 6(b)
    python plot_results_fig34.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```