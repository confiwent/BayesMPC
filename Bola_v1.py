import argparse
import numpy as np
import fixed_env as env
import load_trace


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
MINIMUM_BUFFER_S = 10
BUFFER_TARGET_S = 30

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

parser = argparse.ArgumentParser(description='BOLA')
parser.add_argument('--lin', action='store_true', help='QoE_lin metric')
parser.add_argument('--log', action='store_true', help='QoE_log metric')
parser.add_argument('--FCC', action='store_true', help='Test in FCC dataset')
parser.add_argument('--HSDPA', action='store_true', help='Test in HSDPA dataset')
parser.add_argument('--Oboe', action='store_true', help='Test in Oboe dataset')

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
    Log_file_path = './' + qoe_metric + '/' + dataset + '/log_sim_bola'

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(dataset_path)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    r_batch = []
    gp = 1 - 0 + (np.log(VIDEO_BIT_RATE[-1] / float(VIDEO_BIT_RATE[0])) - 0) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # log
    vp = MINIMUM_BUFFER_S/(0+ gp -1)
    # gp = 1 - VIDEO_BIT_RATE[0]/1000.0 + (VIDEO_BIT_RATE[-1]/1000. - VIDEO_BIT_RATE[0]/1000.) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # lin 
    # vp = MINIMUM_BUFFER_S/(VIDEO_BIT_RATE[0]/1000.0+ gp -1)
    

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

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
        # r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # if buffer_size < RESEVOIR:
        #     bit_rate = 0
        # elif buffer_size >= RESEVOIR + CUSHION:
        #     bit_rate = A_DIM - 1
        # else:
        #     bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        score = -65535
        for q in range(len(VIDEO_BIT_RATE)):
            s = (vp * (np.log(VIDEO_BIT_RATE[q] / float(VIDEO_BIT_RATE[0])) + gp) - buffer_size) / next_video_chunk_sizes[q]
            # s = (vp * (VIDEO_BIT_RATE[q]/1000. + gp) - buffer_size) / next_video_chunk_sizes[q] # lin
            if s>=score:
                score = s
                bit_rate = q

        bit_rate = int(bit_rate)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            print "video count", video_count
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()
