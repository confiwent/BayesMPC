'''without prior knowledge of future bandwidth'''

import argparse
import pdb
import numpy as np
import fixed_env as env
import load_trace
import os

VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300] # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
M_IN_K = 1000.0
# BUFFER_TARGET = 2.5 # in sec
# FOV_W = 0.80
# REBUF_PENALTY = 4.3 # 1 sec rebuffering -> 20 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
# DEFAULT_PSNR = 36
RANDOM_SEED = 42
RAND_RANGE = 1000
# TEST_TRACES = './traces_oboe/'

parser = argparse.ArgumentParser(description='Rate-based')
parser.add_argument('--lin', action='store_true', help='QoE_lin metric')
parser.add_argument('--log', action='store_true', help='QoE_log metric')
parser.add_argument('--FCC', action='store_true', help='Test in FCC dataset')
parser.add_argument('--HSDPA', action='store_true', help='Test in HSDPA dataset')
parser.add_argument('--Oboe', action='store_true', help='Test in Oboe dataset')

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

def main():
    args = parser.parse_args()
    if args.lin:
        qoe_metric = 'results_lin'
    elif args.log:
        qoe_metric = 'results_log'
    else:
        print('Please select the QoE Metric!')
    
    if args.FCC:
        dataset = 'fcc'
    elif args.HSDPA:
        dataset = 'HSDPA'
    elif args.Oboe:
        dataset = 'Oboe'
    else:
        print('Please select the dataset!')
    
    dataset_path = './traces_' + dataset + '/'
    Log_file_path = './' + qoe_metric + '/' + dataset + '/log_sim_rb'

    np.random.seed(RANDOM_SEED)

    # if not os.path.exists(SUMMARY_DIR):
    #     os.makedirs(SUMMARY_DIR)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(dataset_path)
    past_bandwidths = np.zeros(6)
    opt_ptr = 0

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    # current_psnr = DEFAULT_PSNR
    # last_psnr = DEFAULT_PSNR

    video_count = 0

    while True:
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        # throuput_e = np.roll(throuput_e, -1)
        # throuput_e[-1] = float(video_chunk_size) / float(delay) * M_IN_K  # byte/s
        # while throuput_e[0] == 0.0:
        #     throuput_e = throuput_e[1:]
        # bandwidth_sum = 0
        # for past_val in throuput_e:
        #     bandwidth_sum += (1/float(past_val))
        # harmonic_bandwidth = 1.0/(bandwidth_sum/len(throuput_e))
        # throuput_a = harmonic_bandwidth

        past_bandwidths = np.roll(past_bandwidths, -1)
        past_bandwidths[-1] = float(video_chunk_size) / float(delay) * M_IN_K  # byte/s

        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]

        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-past_bandwidths[-1])/float(past_bandwidths[-1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(past_errors) < 5 ):
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)

        chunksize_min = next_video_chunk_sizes[0]

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        if qoe_metric == 'results_lin':
            REBUF_PENALTY = 4.3
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        else:
            REBUF_PENALTY = 2.66
            log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
            log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

            reward = log_bit_rate \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        last_bit_rate = bit_rate
        ## last_psnr = current_psnr
        
        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()
        
        all_reward = []
        all_quality_tuple =[]
        ptr = 0
        # RB-algorithm
        bit_rate = 0
        for q in xrange(5, -1, -1):
            next_size = next_video_chunk_sizes[q]
            if next_size/future_bandwidth - (buffer_size) <= 0:
                bit_rate = q
                break
            #next_psnr = next_chunk_psnr[q]
            # if throuput_a * 2 < next_size:
            #     reward = 0
            # else:
            # reward = VIDEO_BIT_RATE[q] / M_IN_K \
            #             - REBUF_PENALTY * np.maximum(next_size/future_bandwidth - buffer_size, 0) \
            #             - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[q] -
            #                                VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            # log_bit_rate = np.log(VIDEO_BIT_RATE[q] / float(VIDEO_BIT_RATE[0]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

            # reward = log_bit_rate \
            #             - REBUF_PENALTY * np.maximum(next_size/future_bandwidth - buffer_size, 0) \
            #             - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)
            # all_reward.append(reward)
            # all_quality_tuple.append(q)
            # ptr += 1
        # all_reward = np.asarray(all_reward)
        # if all_reward.all() == 0 :
        #     bit_rate = 0
        #     #current_psnr = next_chunk_psnr[bit_rate]
        # else:
        #     opt_ptr = all_reward.argmax()
        #     bit_rate = all_quality_tuple[opt_ptr]
            #current_psnr = next_chunk_psnr[bit_rate]


        if end_of_video:
            log_file.write('\n')
            log_file.close()

            # bit_rate = 0
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            #current_psnr = DEFAULT_PSNR
            del past_bandwidth_ests[:]

            print "video count", video_count
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()









