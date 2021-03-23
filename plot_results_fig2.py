import os
import numpy as np
import matplotlib.pyplot as plt


# RESULTS_FOLDER = './results_backup/traces_02/'
# RESULTS_FOLDER = './results/fcc/'
# RESULTS_FOLDER = './results_lin/fcc/'
# RESULTS_FOLDER = './results/true_belgium/'
# RESULTS_FOLDER = './results/test/'
# RESULTS_FOLDER = './results/unfair_kx/'
# RESULTS_FOLDER = './results/unfair_kx/'
# RESULTS_FOLDER = './results_lin/oboe/'
# RESULTS_FOLDER = './results/norway/'
# RESULTS_FOLDER = './results_lin/model_com/'
# RESULTS_FOLDER = './results_lin/test/'
RESULTS_FOLDER = './results_lin/trace_bw_com/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0 
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired  sequential: 'viridis' cubehelix, jet, rainbow, RdBu, tab20b
SIM_DP = 'sim_dp'
PROPOSED_SCHEME = 'sim_bayesmpc'
PROPOSED_SCHEME_NAME = 'BayesMPC'
#SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP
# SCHEMES = ['sim_bayesmpc', 'sim_bb', 'sim_nrl', 'sim_rb', 'sim_mpc', 'sim_bola'] #, 'sim_pnn3f_mdmpc', 'sim_mpc', 'sim_future7_mpc' , 'sim_pnn3f_mdmpc' 'sim_std3bbpmpc', 'sim_bbpmpc',  'sim_rb', , 'sim_future4_mpc'
# SCHEMES_NAME = ['BayesMPC', 'Buffer-based', 'Pensieve', 'Rate-based', 'RobustMPC', 'BOLA'] # , 'cnnMPC', 'OurProposed-PNN', 'RB', , 'Offline-Optimal'
# SCHEMES = ['sim_bayesmpc', 'sim_mpc', 'sim_10bayesmpc', 'sim_13bayesmpc', 'sim_t1bayesmpc', 'sim_17bayesmpc', 'sim_no_bayesmpc', 'sim_CNNmpc'] #, 'sim_mpc', 'sim_future7_mpc' , 'sim_pnn3f_mdmpc' 'sim_std3bbpmpc', 'sim_bbpmpc', 
# SCHEMES_NAME = ['BayesMPC', 'Robust', 'Bayes(z=1.0)', 'Bayes (z= 1.3)', 'Bayes(z=1.5)', 'Bayes(z=1.7)', 'Bayes(PE)', 'CNN-based'] # , 'OurProposed-PNN'
# SCHEMES = ['sim_bayesmpc', 'sim_t1bayesmpc', 'sim_10bayesmpc', 'sim_PEbayesmpc'] #
# SCHEMES_NAME = ['BayesMPC', 'Bayes(z=1.0)', 'Bayes(z=1.5)', 'Bayes(PE)'] # , 'OurProposed-PNN'
SCHEMES = ['sim_bayesmpc', 'sim_mpc'] #
SCHEMES_NAME = ['BayesMPC', 'RobustMPC'] # , 'OurProposed-PNN'
COLOR_CDF = ['r', '#800080', 'c', '', '', '']
LINE_STY = ['-', ':', '--', '-.', ':', '-.', '-.', '-'] # style set of the lines of figure
HATCH = ['', '/', '+', '\\', '//', '-', 'x']

def Plot_Bar(Plot_value, Y_Label, X_Label, Legend, Legend_column, intra_width = 0.1, intra_interval = 0, inter_width = 0.2, Y_bottom = 0, X_right = 0): # plot a bar, the Plot_value should be a dictionary, the 'Y_Lable' should be a string, the 'X_Label' and 'Legend' should be both a list of strings, 'Legend_column' is the column numbers of the legend, 'intra_width' is the width of a bar on the x-axis, 'intra_interval' is the width of interval of bars on the x-axis in a same term, 'inter_width' is the distance of different terms on the x-axis, 'Y_bottom' is the start value of y-axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # left_start_pos = 0.1
    # intra_width = 0.08
    # inter_width = 0.12
    x_label_pos = []
    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(Legend))]
    
    compare_term_num = len(X_Label) # the number of terms that should be compared
    pos = intra_width * 3 / float(2) # start position
    for term in range(compare_term_num):
        left_start_pos = pos - intra_width/float(2)
        for method in range(len(Legend)):
            # pdb.set_trace()
            plt.bar(pos, Plot_value[term][method], width = intra_width, alpha = 0.5, color = colors[method], edgecolor = 'k', hatch = HATCH[method]) #, hatch = HATCH[method])
            pos += intra_width + intra_interval
        x_label_pos.append(pos - intra_width/float(2) - intra_interval - (pos - intra_width/float(2) - left_start_pos) / float(2))
        pos += inter_width
    
    if Legend[0] != 'None':
        le = ax.legend(Legend, loc = 1, ncol = Legend_column)
        frame = le.get_frame()
        frame.set_alpha(0)
        frame.set_facecolor('none')
    plt.ylabel(Y_Label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle = '--')
    plt.ylim(bottom = Y_bottom)
    ax.set_xticks(x_label_pos)
    ax.set_xticklabels(X_Label)
    # plt.xticks(fontsize = 16)
    # plt.yticks(fontsize = 16)
    if not X_right == 0:
        ax.set_xlim(left = 0, right = X_right)
    plt.show()	

def main():
	time_all = {}
	bit_rate_all = {}
	quality_all = {}
	buff_all = {}
	rebuff_all = {}
	bw_all = {}
	raw_reward_all = {}
	bw_pre_all = {}
	bw_robust_all = {}
	bw_upper_all = {}
	bw_true_all = {}

	for scheme in SCHEMES:
		time_all[scheme] = {}
		raw_reward_all[scheme] = {}
		bit_rate_all[scheme] = {}
		quality_all[scheme] = {}
		buff_all[scheme] = {}
		rebuff_all[scheme] = {}
		bw_all[scheme] = {}
		bw_pre_all[scheme] = {}
		bw_robust_all[scheme] = {}
		bw_upper_all[scheme] = {}
		bw_true_all[scheme] = {}

	log_files = os.listdir(RESULTS_FOLDER)
	for log_file in log_files:

		time_ms = []
		bit_rate = []
		quality = []
		buff = []
		rebuff = []
		bw = []
		reward = []
		bw_pre = []
		bw_robust = []
		bw_upper = []
		bw_true = []

		print(log_file)

		with open(RESULTS_FOLDER + log_file, 'rb') as f:
			if SIM_DP in log_file:
				last_t = 0
				last_b = 0
				last_q = 1
				lines = []
				for line in f:
					lines.append(line)
					parse = line.split()
					if len(parse) >= 6:
						time_ms.append(float(parse[3]))
						bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
						buff.append(float(parse[4]))
						bw.append(float(parse[5]))
				
				for line in reversed(lines):
					parse = line.split()
					r = 0
					if len(parse) > 1:
						if int(parse[0]) < 48:
							t = float(parse[3])
							b = float(parse[4])
							q = int(parse[6])
							if b == 4:
								rebuff = (t - last_t) - last_b
								assert rebuff >= -1e-4
								r -= REBUF_P * rebuff

							r += VIDEO_BIT_RATE[q] / K_IN_M
							r -= SMOOTH_P * np.abs(VIDEO_BIT_RATE[q] - VIDEO_BIT_RATE[last_q]) / K_IN_M
							reward.append(r)

							last_t = t
							last_b = b
							last_q = q

			else:
				for line in f:
					parse = line.split()
					if len(parse) <= 1:
						break
					time_ms.append(float(parse[0]))
					bit_rate.append(int(parse[1]))
					quality.append(float(parse[1])/1000.)
					# quality.append(float(np.log(float(parse[1]) / float(VIDEO_BIT_RATE[0])))
					buff.append(float(parse[2]))
					rebuff.append(float(parse[3]))
					bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
					reward.append(float(parse[6]))
					bw_pre.append(float(parse[7]))
					bw_robust.append(float(parse[9]))
					bw_upper.append(float(parse[7])+ float(parse[8]))
					bw_true.append(float(parse[10]))

		if SIM_DP in log_file:
			time_ms = time_ms[::-1]
			bit_rate = bit_rate[::-1]
			buff = buff[::-1]
			bw = bw[::-1]
		
		time_ms = np.array(time_ms)
		time_ms -= time_ms[0]
		
		# print log_file

		for scheme in SCHEMES:
			if scheme in log_file:
				time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
				bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
				quality_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = quality
				buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
				rebuff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = rebuff
				bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
				raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
				bw_pre_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw_pre
				bw_robust_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw_robust
				bw_upper_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw_upper
				bw_true_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw_true
				break

	# ---- ---- ---- ----bayesmpc
	# Reward records
	# ---- ---- ---- ----
		
	log_file_all = []
	reward_all = {}
	quality_Mtrace = {}
	rebuff_Mtrace = {}
	smooth_Mtrace = {}
	mse_bwP_Mtrace = {}
	er_bwP_Mtraces = {}
	reward_improvement = {}
	for scheme in SCHEMES:
		reward_all[scheme] = []
		quality_Mtrace[scheme] = []
		rebuff_Mtrace[scheme] = []
		smooth_Mtrace[scheme] = []
		mse_bwP_Mtrace[scheme] = []
		er_bwP_Mtraces[scheme] = []
		if scheme != PROPOSED_SCHEME:
			reward_improvement[scheme]=[]

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			log_file_all.append(l)
			for scheme in SCHEMES:
				reward_all[scheme].append(np.mean(raw_reward_all[scheme][l][10:VIDEO_LEN]))
				quality_Mtrace[scheme].append(np.mean(quality_all[scheme][l][10:VIDEO_LEN]))
				rebuff_Mtrace[scheme].append(np.mean(rebuff_all[scheme][l][10:VIDEO_LEN]))
				#calculate the quality smoothness 
				quality_trace = quality_all[scheme][l][10:VIDEO_LEN]
				quality_last_trace = np.roll(quality_trace, 1)
				quality_last_trace[0] = quality_trace[9] # gurantee the first chunk's smooth penality
				trace_smoothness = [np.abs(quality_trace[ptr] - quality_last_trace[ptr]) for ptr in range(len(quality_trace))]
				smooth_Mtrace[scheme].append(np.mean(trace_smoothness))
				bw_pre_trace = bw_pre_all[scheme][l][10:VIDEO_LEN]
				bw_true_trace = bw_true_all[scheme][l][10:VIDEO_LEN]
				trace_bwmse = [(bw_pre_trace[ptr] -  bw_true_trace[ptr]) for ptr in range(len(bw_pre_trace))]
				mse_bwP_Mtrace[scheme].append(np.mean(trace_bwmse))
				bw_robust_trace = bw_robust_all[scheme][l][10:VIDEO_LEN]
				trace_eratio = [int(bw_robust_trace[ptr]/bw_true_trace[ptr]) for ptr in range(len(bw_robust_trace))] # 1 if bw_robust is greater than true, else 0
				er_bwP_Mtraces[scheme].append(np.mean(trace_eratio)) ## record the error ratio
				# traces_mean = np.mean()
			
	for l in range(len(reward_all[PROPOSED_SCHEME])):
		comparison_schemes = [SCHEMES[i] for i in range(len(SCHEMES))]
		comparison_schemes.remove(PROPOSED_SCHEME)
		for scheme in comparison_schemes:
			reward_improvement[scheme].append(float((reward_all[PROPOSED_SCHEME][l] - reward_all[scheme][l])/1.0 )) # abs(reward_all[scheme][l])


	# for l in time_all[SCHEMES[0]]:
	# 	if np.sum(raw_reward_all['sim_mpc'][l][1:VIDEO_LEN]) != np.sum(raw_reward_all['sim_tmpc'][l][1:VIDEO_LEN]):
	# 		print(str(l))

	mean_rewards = {}
	mean_bw_mse = {}
	mean_bw_er = {}
	mean_quality = []
	mean_rebuffer = []
	mean_smoothness = []
	for scheme in SCHEMES:
		mean_rewards[scheme] = np.mean(reward_all[scheme])
		mean_bw_mse[scheme] = np.mean(mse_bwP_Mtrace[scheme])
		mean_bw_er[scheme] = np.mean(er_bwP_Mtraces[scheme])
		mean_quality.append(np.mean(quality_Mtrace[scheme]))
		mean_rebuffer.append(np.mean(rebuff_Mtrace[scheme]) * REBUF_P)
		mean_smoothness.append(np.mean(smooth_Mtrace[scheme]) * SMOOTH_P)

        # ## calculate the empirical risk
        # empiric_risk = {}
        # near_optimal = reward_all['sim_future5_mpc']
        # for scheme in SCHEMES:
        #     square_error = 0
        #     for index in range(len(near_optimal)):
        #         square_error += (near_optimal[index] - reward_all[scheme][index])**2

        #     empiric_risk[scheme] = square_error

            # empiric_risk[scheme] = square_error

	# fig = plt.figure()
	# ax = fig.add_subplot(111)

	# for scheme in SCHEMES:
	# 	ax.plot(reward_all[scheme])
	
	# SCHEMES_REW = []
	# index = 0
	# for scheme in SCHEMES:
	# 	# SCHEMES_REW.append(SCHEMES_NAME[index] + ': ' + str('%.3f' % mean_rewards[scheme]))
	# 	SCHEMES_REW.append(SCHEMES_NAME[index])
	# 	index += 1

	# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	# for i,j in enumerate(ax.lines):
	# 	j.set_color(colors[i])

	# ax.legend(SCHEMES_REW, loc='best')
	
	# plt.ylabel('total reward')
	# plt.xlabel('trace index')
	# ax.spines['bottom'].set_linewidth(2.5)
	# ax.spines['left'].set_linewidth(2.5)
	# plt.grid()
	# plt.show()

	# # ---- ---- ---- ----
	# # CDF 
	# # ---- ---- ---- ----

	# fig = plt.figure()
	# ax = fig.add_subplot(111)

	# # for method in METHODS:
    # #     values, base = np.histogram(Method_total_qoe_trace[method], bins = NUM_BINS)
    # #     cumulative = np.cumsum(values) / float(len(Method_total_qoe_trace[method]))
    # #     cdf_m.plot(base[:-1], cumulative)
    
    # # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(METHODS))]
    # # for i,j in enumerate(cdf_m.lines):
    # #     # j.set_color(colors[i])
    # #     plt.setp(j, color = colors[i], linestyle = LINE_STY[i], linewidth = 2.0) #, marker = HATCH[i]
    
    # # cdf_m.legend(METHODS_LABEL, loc = 'best', fontsize = 16)

    # # plt.ylabel('CDF', fontsize = 16)
    # # plt.xlabel("Average Values of Chunk's QoE", fontsize = 16)
    # # # cdf_m.set_xlabel(u'时间切片的QoE平均?, fontproperties=font)
    # # plt.xticks(fontsize = 14)
    # # plt.yticks(fontsize = 14)
    # # plt.xlim([0.930,0.994])
    # # plt.title('BC')
    # # plt.grid()
    # # plt.show()

	# for scheme in SCHEMES:
	# 	values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
	# 	cumulative = np.cumsum(values)/float(len(reward_all[scheme]))
	# 	ax.plot(base[:-1], cumulative)	

	# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	# for i,j in enumerate(ax.lines):
	# 	# j.set_color(colors[i])	
	# 	plt.setp(j, color = colors[i], linestyle = LINE_STY[i], linewidth = 2.6) #, marker = HATCH[i]

	# legend = ax.legend(SCHEMES_REW, loc=4, fontsize = 16)
	# frame = legend.get_frame()
	# frame.set_alpha(0)
	# frame.set_facecolor('none')
	# plt.ylabel('CDF (Perc. of sessions)', fontsize = 20)
	# plt.xlabel("Average Values of Chunk's $QoE_{lin}$", fontsize = 18)
	# plt.xticks(fontsize = 16)
	# plt.yticks(fontsize = 16)
	# # plt.xlim([0.930,0.994])
	# ax.spines['top'].set_visible(False)
	# ax.spines['right'].set_visible(False)
	# ax.spines['bottom'].set_linewidth(2.5)
	# ax.spines['left'].set_linewidth(2.5)
	# plt.title('HSDPA') # HSDPA , FCC , Oboe
	# # plt.grid()
	# plt.show()

	# #################################################################
	# # QoE reward_improvement
	# #################################################################

	# fig = plt.figure()
	# ax = fig.add_subplot(111)

	# for scheme in comparison_schemes:
	# 	values, base = np.histogram(reward_improvement[scheme], bins=NUM_BINS)
	# 	cumulative = np.cumsum(values)/float(len(reward_improvement[scheme]))
	# 	ax.plot(base[:-1], cumulative)	

	# # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	# for i,j in enumerate(ax.lines):
	# 	# j.set_color(colors[i])	
	# 	plt.setp(j, color = colors[i+1], linestyle = LINE_STY[i], linewidth = 2.6) #, marker = HATCH[i]

	# comparison_schemes_names = [SCHEMES_NAME[i] for i in range(len(SCHEMES_NAME))]
	# comparison_schemes_names.remove(PROPOSED_SCHEME_NAME)

	# legend = ax.legend(comparison_schemes_names, loc='best', fontsize = 17)
	# 	# legend = ax.legend(SCHEMES_REW, loc=4, fontsize = 14)
	# frame = legend.get_frame()
	# frame.set_alpha(0)
	# frame.set_facecolor('none')
	# plt.ylabel('CDF (Perc. of sessions)', fontsize = 20)
	# plt.xlabel("$QoE_{lin}$ improvement", fontsize = 18)
	# plt.xticks(fontsize = 16)
	# plt.yticks(fontsize = 16)
	# plt.ylim([0.0,1.0])
	# # plt.xlim(-0.2, 1)
	# plt.vlines(0, 0, 1, colors=colors[0],linestyles='solid')
	# ax.spines['top'].set_visible(False)
	# ax.spines['right'].set_visible(False)
	# ax.spines['bottom'].set_linewidth(2.5)
	# ax.spines['left'].set_linewidth(2.5)
	# plt.title('HSDPA') # HSDPA , FCC , Oboe
	# # plt.grid()
	# plt.show()

	# #################################################################
	# # QoE reward_improvement
	# #################################################################
	# Plot_Bar_value = {}
	# Plot_Bar_value[0] = [np.log(1+i) for i in mean_quality]
	# Plot_Bar_value[1] = [np.log(1+i) for i in mean_rebuffer]
	# Plot_Bar_value[2] = [np.log(1+i) for i in mean_smoothness]
	# Plot_Bar(Plot_Bar_value, "Average value", ['Bitrate utility', 'Rebuffering penalty', 'Smoothness penalty'], SCHEMES_NAME, 2, 0.2, 0, 0.2, X_right = 4.4)

	# ---- ---- ---- ----
	# check each trace
	# ---- ---- ---- ----

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][1:VIDEO_LEN], bw_robust_all[scheme][l][1:VIDEO_LEN])
			scheme = PROPOSED_SCHEME
			ax.plot(time_all[scheme][l][1:VIDEO_LEN], bw_pre_all[scheme][l][1:VIDEO_LEN])
			ax.plot(time_all[scheme][l][1:VIDEO_LEN], bw_true_all[scheme][l][1:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				# j.set_color(colors[i])	# 
				plt.setp(j, color = colors[i], linestyle = LINE_STY[i], linewidth = 2.6) #, marker = HATCH[i]
			plt.fill_between(time_all[scheme][l][1:VIDEO_LEN], bw_robust_all[scheme][l][1:VIDEO_LEN], bw_upper_all[scheme][l][1:VIDEO_LEN], color = 'g', alpha = 0.2, label = 'uncertainty')
			plt.title(l, fontsize = 16)
			# plt.ylabel('Throughputs Prediction (MBps)')
			SCHEMES_REW = ['Lower bound of BayesMPC', 'Lower bound of RobustMPC', 'Mean value of BayesMPC', 'True']
			legend = ax.legend(SCHEMES_REW, loc=1, fontsize = 15)
			frame = legend.get_frame()
			frame.set_alpha(0)
			frame.set_facecolor('none')
			plt.ylabel('Throughputs Prediction (MBps)', fontsize = 20)
			plt.xlabel("Time (s)", fontsize = 20)
			plt.xticks(fontsize = 16)
			plt.yticks(fontsize = 16)
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['bottom'].set_linewidth(2.5)
			ax.spines['left'].set_linewidth(2.5)
			plt.show()

			# fig = plt.figure()

			# ax = fig.add_subplot(311)
			# for scheme in SCHEMES:
			# 	ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
			# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			# for i,j in enumerate(ax.lines):
			# 	j.set_color(colors[i])	
			# plt.title(l)
			# plt.ylabel('bit rate selection (kbps)')

			# ax = fig.add_subplot(312)
			# for scheme in SCHEMES:
			# 	ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
			# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			# for i,j in enumerate(ax.lines):
			# 	j.set_color(colors[i])	
			# plt.ylabel('buffer size (sec)')

			# ax = fig.add_subplot(313)
			# for scheme in SCHEMES:
			# 	ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
			# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			# for i,j in enumerate(ax.lines):
			# 	j.set_color(colors[i])	
			# plt.ylabel('bandwidth (mbps)')
			# plt.xlabel('time (sec)')

			# SCHEMES_REW = []
			# for scheme in SCHEMES:
			# 	SCHEMES_REW.append(scheme + ': ' + str('%.3f' %np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))

			# ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.3), ncol=int(np.ceil(len(SCHEMES) / 3.0)))
			# plt.show()


if __name__ == '__main__':
	main()
