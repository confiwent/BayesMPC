import argparse
import numpy as np
import fixed_env as env
import load_trace
import matplotlib.pyplot as plt
import itertools
import time


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 3
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
VIDEO_SIZE_FILE = './video_size/ori/video_size_' 
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

parser = argparse.ArgumentParser(description='RobustMPC')
parser.add_argument('--lin', action='store_true', help='QoE_lin metric')
parser.add_argument('--log', action='store_true', help='QoE_log metric')
parser.add_argument('--FCC', action='store_true', help='Test in FCC dataset')
parser.add_argument('--HSDPA', action='store_true', help='Test in HSDPA dataset')
parser.add_argument('--Oboe', action='store_true', help='Test in Oboe dataset')
parser.add_argument('--cb', action='store_true', help='Compare the lower bound')

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

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
    args = parser.parse_args()
    if args.cb or args.lin:
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
    if args.cb:
        Log_file_path = './' + qoe_metric + '/cb_' + dataset + '/log_sim_mpc'
    else:
        Log_file_path = './' + qoe_metric + '/' + dataset + '/log_sim_mpc'
    
    start = time.time()

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(dataset_path)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    chunk_size_info = video_size()
    chunk_size_info.store_size()

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    harmonic_bandwidth = 0 
    future_bandwidth = 0

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    # entropy_record = []

    video_count = 0
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
        if qoe_metric == 'results_lin':
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
                       str(harmonic_bandwidth) + '\t' + 
                       str(harmonic_bandwidth - future_bandwidth) + '\t' + 
                       str(future_bandwidth) + '\t' +
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
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
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


        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain -1)
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
                download_time = (chunk_size_info.get_chunk_size(action, index)/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
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
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

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

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                end = time.time()
                print(end - start)
                break

            log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')

            end = time.time()
            print(end - start)


if __name__ == '__main__':
    main()

