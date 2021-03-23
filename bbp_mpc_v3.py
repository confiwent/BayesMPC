''' 
use pnn to predict the real values of states
using uncetainty 
the input length of Bayes is 10 (6 is worse than 10 and 8)
For planning, the node pruning method is employed in the decision-making tree

'''
import numpy as np
import fixed_env as env
import load_trace
import matplotlib.pyplot as plt
import itertools
# import bandwidth_pred as bp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
# import test as bw_pre
# import mcts_abr_pnn as mcts
import pandas
from collections import Counter
# import lstm_net as net


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 10  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 3
SIMULATION_NUM = 8
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000

# QOE_METRIC = 'results_lin' # QoE_lin
QOE_METRIC = 'results_log' # QoE_log
# DATASET  = 'HSDPA' # HSDPA
DATASET  = 'fcc' # HSDPA

DATA_SET_PATH = './traces_' + DATASET + '/'
SUMMARY_DIR = './' + QOE_METRIC + '/' + DATASET
LOG_FILE = './' + QOE_METRIC + '/' + DATASET + '/log_sim_bayesmpc'
VIDEO_SIZE_FILE = './video_size/ori/video_size_'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

TIME_STEP = 8
# TARGET_SIZE = 5
INPUT_SIZE = 1
HIDDEN_SIZE = 128

BATCH_SIZE = 1
LR = 0.001
EPOCH = 20

CHUNK_COMBO_OPTIONS = []


# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

#size_video1 = [3155849, 2641256, 2410258, 2956927, 2593984, 2387850, 2554662, 2964172, 2541127, 2553367, 2641109, 2876576, 2493400, 2872793, 2304791, 2855882, 2887892, 2474922, 2828949, 2510656, 2544304, 2640123, 2737436, 2559198, 2628069, 2626736, 2809466, 2334075, 2775360, 2910246, 2486226, 2721821, 2481034, 3049381, 2589002, 2551718, 2396078, 2869088, 2589488, 2596763, 2462482, 2755802, 2673179, 2846248, 2644274, 2760316, 2310848, 2647013, 1653424]
# size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
# size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
# size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
# size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
# size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
# size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

# def create_DataTensor(dataset, window):
#     # Input: dataset, flattened numpy array
#     # Output: Tensor_x [batch, time_step, features]
#     #         Tensor_y [batch, target_size, features]
#     # Each dataset has one output
#     data_x, data_y = [], []
#     for i in range(window, len(dataset)):
#         batch_x = dataset[i - window: i]
#         batch_y = dataset[i]
#         data_x.append(batch_x[:, np.newaxis])
#         data_y.append(np.array([[batch_y]]))

#     data_x, data_y = np.asarray(data_x), np.asarray(data_y)

#     Tensor_x = torch.from_numpy(data_x)
#     Tensor_y = torch.from_numpy(data_y)
#     return Tensor_x, Tensor_y


# torch.cuda.device(0)
# torch.cuda.get_device_name(torch.cuda.current_device())


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out

def log_gaussian_loss(output, target, sigma, no_dim, sum_reduce=True):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    
    if sum_reduce:
        return -(log_coeff + exponent).sum()
    else:
        return -(log_coeff + exponent)


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)
    
    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()
    
    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def loglik(self, weights): # log Guassian value 
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
        return (exponent + log_coeff).sum()

class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        
        # scale = (2/self.input_dim)**0.5
        # rho_init = np.log(np.exp((2/self.input_dim)**0.5) - 1)
        self.weight_mus = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.weight_rhos = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -3))
        
        self.bias_mus = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -3))
        
    def forward(self, x, sample = True):
        
        if sample:
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            bias_epsilons =  Variable(self.bias_mus.data.new(self.bias_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            bias_stds = torch.log(1 + torch.exp(self.bias_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            bias_sample = self.bias_mus + bias_epsilons*bias_stds
            
            output = torch.mm(x, weight_sample) + bias_sample ## Performs a matrix multiplication of the matrices input and mat2.
            
            # computing the KL loss term
            prior_cov, varpost_cov = self.prior.sigma**2, weight_stds**2
            KL_loss = 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*weight_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.weight_mus - self.prior.mu)**2/prior_cov).sum()
            
            prior_cov, varpost_cov = self.prior.sigma**2, bias_stds**2
            KL_loss = KL_loss + 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*bias_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.bias_mus - self.prior.mu)**2/prior_cov).sum()
            
            return output, KL_loss
        
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, KL_loss
        
    def sample_layer(self, no_samples):
        all_samples = []
        for i in range(no_samples):
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            
            all_samples += weight_sample.view(-1).cpu().data.numpy().tolist()
            
        return all_samples


class BBP_Heteroscedastic_Model_Wrapper:
    def __init__(self, network, learn_rate, batch_size, no_batches):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches
        
        self.network = network
        self.network.cuda()
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learn_rate)
        self.loss_func = log_gaussian_loss
    
    def fit(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=True)
        
        # reset gradient and total loss
        self.optimizer.zero_grad()
        fit_loss_total = 0
        
        for i in range(no_samples):
            output, KL_loss_total = self.network(x)

            # calculate fit loss based on mean and standard deviation of output
            fit_loss = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1) ## sigma.exp() make sure the sigma > 0
            fit_loss_total = fit_loss_total + fit_loss
        
        KL_loss_total = KL_loss_total/self.no_batches
        total_loss = (fit_loss_total + KL_loss_total)/(no_samples*x.shape[0])
        total_loss.backward()
        self.optimizer.step()

        return fit_loss_total/no_samples, KL_loss_total
    
    def get_loss_and_rmse(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=True)
        
        means, stds = [], []
        for i in range(no_samples):
            output, KL_loss_total = self.network(x)
            means.append(output[:, :1, None])
            stds.append(output[:, 1:, None].exp())
            
        means, stds = torch.cat(means, 2), torch.cat(stds, 2)
        mean = means.mean(dim=2)
        std = (means.var(dim=2) + stds.mean(dim=2)**2)**0.5
            
        # calculate fit loss based on mean and standard deviation of output
        logliks = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1, sum_reduce=False)
        rmse = float((((mean - y)**2).mean()**0.5).cpu().data)

        return logliks, rmse

class BBP_Heteroscedastic_Model_UCI(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(BBP_Heteroscedastic_Model_UCI, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, num_units, gaussian(0, 1))
        self.layer2 = BayesLinear_Normalq(num_units, num_units, gaussian(0, 1))
        self.layer3 = BayesLinear_Normalq(num_units, 2*output_dim, gaussian(0, 1))
        
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace = True)
    
    def forward(self, x):
        
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)
        
        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)
        
        x, KL_loss = self.layer3(x)
        KL_loss_total = KL_loss_total + KL_loss
        
        return x, KL_loss_total

def throughput_pre(data, model, horizon):
    # best_net = torch.load("./throughput_v0.pkl")
    best_net = model
    H_step = horizon
    in_dim = 10
    x_test = data
    throughput_mean = []
    throughput_std = []
    x, y = to_variable(var=(x_test, x_test), cuda=True)

    for step in range(H_step):
        means, stds = [], []
        no_samples = SIMULATION_NUM
        for i in range(no_samples):
            output, KL_loss_total = best_net(x)
            means.append(output[:, :1, None])
            stds.append(output[:, 1:, None].exp())
            
        means, stds = torch.cat(means, 2), torch.cat(stds, 2)
        mean = means.mean(dim=2)
        std = (means.var(dim=2) + stds.mean(dim=2)**2)**0.5
        # std = stds.mean(dim=2)
        throughput_mean.append(mean.cpu().detach().numpy()[0].tolist()[0])
        throughput_std.append(std.cpu().detach().numpy()[0].tolist()[0])

        x_test = np.roll(x_test, -1, axis=0)
        x_test[-1] = mean.cpu().detach().numpy()
        x, y = to_variable(var=(x_test, x_test), cuda=True)


    return throughput_mean, throughput_std


# def get_chunk_size(quality, index):
#     if ( index < 0 or index > 48 ):
#         return 0
#     # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
#     sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
#     return sizes[quality]
class video_size(object):
    def __init__(self):
        self.video_sizes = {}

    def store_size(self):
        for bitrate in range(A_DIM):
            self.video_sizes[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_sizes[bitrate].append(int(line.split()[0]))

    def get_chunk_size(self, quality, index):
        if ( index < 0 or index > 47 ):
            return 0
        # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
        # sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
        return self.video_sizes[quality][index]

def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(DATA_SET_PATH)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    model = torch.load("./throughput_v0.pkl")

    chunk_size_info = video_size()
    chunk_size_info.store_size()

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    future_bandwidth = [0]
    future_bandwidth_r = [0]
    predicted_mean = [0]
    predicted_unc = [0]
    # entropy_record = []

    video_count = 0

    # make chunk combination options
    # for combo in itertools.product([0,1,2,3,4,5], repeat=MPC_FUTURE_CHUNK_COUNT):
    #     CHUNK_COMBO_OPTIONS.append(combo)

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, _,\
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        if QOE_METRIC == 'results_lin':
            REBUF_PENALTY = 4.3
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        else:# log scale reward
            REBUF_PENALTY = 2.66
            log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
            log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

            reward = log_bit_rate \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # reward = BITRATE_REWARD[bit_rate] \
        #          - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[last_bit_rate])


        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\t' + 
                       str(predicted_mean[0]/8) + '\t' + str(predicted_unc[0]/8) + '\t' + 
                       str(future_bandwidth_r[0]/8) + '\t' + 
                       str(float(video_chunk_size) / float(delay) / M_IN_K) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K * 8# kilo bits / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        # curr_error_for_net = 0
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        # curr_error_for_net = past_bandwidth_ests[-1]-state[3,-1]
        past_errors.append(curr_error)

    
        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        # past_bandwidths = [0 for i in range(10)]
        past_bandwidths = state[3,-S_LEN:]

        # simulation for many times
        est_current = []
        predicted_mean, predicted_unc = throughput_pre(past_bandwidths, model, MPC_FUTURE_CHUNK_COUNT)
        future_bandwidth = predicted_mean
        # last_bandwidth = state[3, -1]
        # for i in range(MPC_FUTURE_CHUNK_COUNT):
        #     future_bandwidth.append(last_bandwidth + prediction_output[i])
        #     last_bandwidth = last_bandwidth + prediction_output[i]
        # while past_bandwidths[0] == 0.0:
        #     past_bandwidths = past_bandwidths[1:]
        # #if ( len(state) < 5 ):
        # #    past_bandwidths = state[3,-len(state):]
        # #else:
        # #    past_bandwidths = state[3,-5:]
        # bandwidth_sum = 0
        # for past_val in past_bandwidths:
        #     bandwidth_sum += (1/float(past_val))
        # harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(past_errors) < 5 ):
            error_pos = -len(past_errors)
        max_error = float(np.max(past_errors[error_pos:]))
        future_bandwidth_r = []
        for i in range(MPC_FUTURE_CHUNK_COUNT): #MPC_FUTURE_CHUNK_COUNT
            # future_bandwidth_r.append(future_bandwidth[i]/(1+max_error))  # robustMPC here
            future_bandwidth_r.append(max(future_bandwidth[i] - 1.2*predicted_unc[i], 0.01))  # robustMPC hered
            # future_bandwidth_r.append(max(future_bandwidth[i] - (2.2-np.log10(buffer_size))*predicted_unc[i], 0.001))
        est_current.append(future_bandwidth[0])
        past_bandwidth_ests.append(future_bandwidth[0])


        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if ( TOTAL_VIDEO_CHUNKS - last_index < MPC_FUTURE_CHUNK_COUNT ):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        # best_combo = ()
        start_buffer = buffer_size
        #start = time.time()
        download_time_every_step = []
        for position in range(future_chunk_length):
            download_time_current = []
            for action in range(0, A_DIM):
                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (chunk_size_info.get_chunk_size(action, index)/1000000.)/future_bandwidth_r[0] * 8 # this is MB/MB/s --> seconds
                download_time_current.append(download_time)
            download_time_every_step.append(download_time_current)

        reward_comparison = False
        send_data = 0
        parents_pool = [[0.0, start_buffer, int(bit_rate)]]
        for position in range(future_chunk_length):
            if position == future_chunk_length-1:
                reward_comparison = True
            children_pool = []
            for parent in parents_pool:
                action = 0
                curr_buffer = parent[1]
                last_quality = parent[-1]
                curr_rebuffer_time = 0
                chunk_quality = action
                download_time = download_time_every_step[position][chunk_quality]
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0.0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                # reward
                bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (SMOOTH_PENALTY*smoothness_diffs/1000.)
                reward += parent[0]

                children = parent[:]
                children[0] = reward
                children[1] = curr_buffer
                children.append(action)
                children_pool.append(children)
                if (reward >= max_reward) and reward_comparison:
                    if send_data > children[3] and reward == max_reward:
                        send_data = send_data
                    else:
                        send_data = children[3]
                    max_reward = reward

                # criterion terms
                # theta = SMOOTH_PENALTY * (VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)
                rebuffer_term = REBUF_PENALTY * (max(download_time_every_step[position][action+1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0))
                if (action + 1 <= parent[-1]):
                    High_Maybe_Superior = ((1.0 + 2 * SMOOTH_PENALTY)*(VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0.0)
                else:
                    High_Maybe_Superior = ((VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0.0)



                # while REBUF_PENALTY*(download_time_every_step[position][action+1] - parent[1]) <= ((VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)-(abs(VIDEO_BIT_RATE[action+1] - VIDEO_BIT_RATE[parent[-1]]) - abs(VIDEO_BIT_RATE[action] - VIDEO_BIT_RATE[parent[-1]]))/1000.):
                while High_Maybe_Superior:
                    curr_buffer = parent[1]
                    last_quality = parent[-1]
                    curr_rebuffer_time = 0
                    chunk_quality = action + 1
                    download_time = download_time_every_step[position][chunk_quality]
                    if ( curr_buffer < download_time ):
                        curr_rebuffer_time += (download_time - curr_buffer)
                        curr_buffer = 0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += 4

                    # reward
                    bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                    reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (SMOOTH_PENALTY*smoothness_diffs/1000.)
                    reward += parent[0]

                    children = parent[:]
                    children[0] = reward
                    children[1] = curr_buffer
                    children.append(chunk_quality)
                    children_pool.append(children)
                    if (reward >= max_reward) and reward_comparison:
                        if send_data > children[3] and reward == max_reward:
                            send_data = send_data
                        else:
                            send_data = children[3]
                        max_reward = reward

                    action += 1
                    if action + 1 == A_DIM:
                        break
                    # criterion terms
                    # theta = SMOOTH_PENALTY * (VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)
                    rebuffer_term = REBUF_PENALTY * (max(download_time_every_step[position][action+1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0))
                    if (action + 1 <= parent[-1]):
                        High_Maybe_Superior = ((1.0 + 2 * SMOOTH_PENALTY)*(VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0)
                    else:
                        High_Maybe_Superior = ((VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0)

            parents_pool = children_pool
        bit_rate = send_data
        # for item in L:
        #     if L[item] > max_time:
        #         max_time = L[item]
        #         bit_rate = item
        #     elif L[item] == max_time:
        #         if bit_rate < item:
        #             bit_rate = item
        # bit_rate = send_data
        # hack
        # if bit_rate == 1 or bit_rate == 2:s
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)
        past_bandwidth_ests.append(np.mean(est_current))

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            del past_bandwidth_ests[:]

            time_stamp = 0

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []
            # past_errors_for_net = []

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()

