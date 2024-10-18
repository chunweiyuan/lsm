NEURONAL_PARAMS = {'exc_w':10.0, 
                   'inh_w':-10.0, 
                   'syn_delay':1.0,
                   'tau_m':30.0, 
                   'r_inp':10.0, 
                   'tau_syn_ex':10.0, 
                   'tau_syn_in':10.0, 
                   't_hyp':10.0, 
                   't_hyp_std':0.0, 
                   'V_hyp_0':-10.0, 
                   'V_hyp_0_std':0.0}

NETWORK_PARAMS = {'n_total':2, 
                  'n_inputs':1, 
                  'inp_conn':1.0,
                  'p_exc':1.0, 
                  'p_connect_ran':0.0}

STIMULUS_PARAMS = {'burst_duration': 100.0,
                   'burst_freq': 100.0,
                   'gap_duration': 100.0,
                   'anomaly_duration': 50.0,
                   'anomaly_freq': 1000.0,
                   'anomaly_fraction': 0.2,
                   'description': 'blah blah',
                   'n_patterns': 10,
                   'n_fibers': 2,
                   'randomize': True}

TXTSET_PARAMS = {'n': 10} # number of pattern repetitions

SIMULATION_PARAMS = {'number_of_runs':100}
