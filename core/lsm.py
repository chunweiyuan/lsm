# This is my attempt to construct the simplest liquid state machine (LSM)
# In reality what I have here is quite a simple recurrent neural network
# Based on InF neurons from NEST Initiative

import nest  # assume nest is installed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")  # MacOSX
import random
import numpy as np
import nest.voltage_trace
import nest.raster_plot
import sys
import pickle
import numpy.linalg
from poisson_spikes import poisson_spikes, multiple_trains
from stimulus import *


class LSM(object):

    def __init__(self, n_total, n_inputs, inp_conn,
                 p_exc, p_connect_ran,
                 exc_w, inh_w,
                 syn_delay,
                 tau_m, r_inp,
                 tau_syn_ex, tau_syn_in,
                 t_hyp, t_hyp_std,
                 V_hyp_0, V_hyp_0_std):
        """

        Args:
            n_total (int): number of liquid neurons
            n_inputs (int): number of input neurons projecting into the liquid
            p_exc (int): fraction of network that is excitatory
            p_connect_ran (float): the random connectivity [0, 1]
            exc_w (float): excitatory weight (mV)
            inh_w (float): inhibitory weight (mV)
            syn_delay (float): synaptic delay (ms)
            tau_m (float): membrane time constant (ms)
            r_inp (float): input resistance (Giga-ohm)
            tau_syn_ex (float): excitatory post-synaptic current time constant (ms)
            tau_syn_in (float): inhibitory post-synaptic current time constant (ms)
            t_hyp (float): hyperpolarization time constant (ms)
            t_hyp_std (float): std dev of t_hyp
            V_hyp_0 (float): initial hyperpolarization magnitude
            V_hyp_0_std (float): std dev of V_hyp_0
        """
        # set neuronal parameters
        self.n_total, self.n_inputs = (n_total, n_inputs)
        self.inp_conn, self.p_exc, self.p_connect_ran = (inp_conn, p_exc, p_connect_ran)
        self.exc_w, self.inh_w = (exc_w, inh_w)
        self.delay = syn_delay # in ms
        self.r_inp = r_inp  # input resistance
        self.tau_m = tau_m  # membrance time constant, in ms
        self.tau_syn_ex, self.tau_syn_in = (tau_syn_ex, tau_syn_in)  # time constants of exc/inh post-synaptic currents
        self.t_hyp, self.V_hyp_0 = (t_hyp, V_hyp_0) # hyperpolarization parameters
        # Make sure your membrane time constant is never equal either tau_psc_ex or tau_psc_in
        assert self.tau_m != self.tau_syn_ex
        assert self.tau_m != self.tau_syn_in
        self.set_neuronal_defaults()
        self.initialize_network_template()
        self.initialize_input_template() # input fibers/connectivity must also be initialized
        self.set_neuronal_parameters(t_hyp_std, V_hyp_0_std)
        self.random_network_connect(exclude_self = True)
        self.set_experiment()
        print 'done setting up network'


    def set_neuronal_defaults(self):
        """
        sets neuronal defaults
        """
        self.time_res = 0.1   # ms
        self.I_e = 0.0        # constant background current (pA) pico-ampere
        self.V_th = -55.0     # firing threshold voltage (mV)
        self.V_reset = -70.0  # reset voltage (mV)
        self.E_L = -70.0      # resting potential (mV)
        self.t_ref = 2.0      # refractory time (ms)


    def initialize_network_template(self):
        """
        Initializes a bunch of empty place holders for network parameters
        """
        # Initialize the indices of network neurons as place-holders.
        self.liquid = np.arange(self.n_total)+1
        # define exc. and inh. neurons
        self.group_id = np.zeros(self.n_total, dtype=int) # initialize all as exc. neurons first
        self.group_id[int(self.n_total*self.p_exc):self.n_total] = 1 # this sets the inh. neurons to id = 1
        self.exc_group = np.array(self.liquid)[self.group_id==0].tolist()
        self.inh_group = np.array(self.liquid)[self.group_id==1].tolist()


    def initialize_input_template(self):
        """
        initializes input connectivity.
        """
        N = len(self.liquid)
        self.input_connect_list = []
        for i in range(self.n_inputs): # enumerate over input trains
            # the +1 is there because "0" is the mother node
            proj = np.random.permutation(np.arange(N)+1)[range(int(N*self.inp_conn))]
            # this just says what liquid neurons this particular input train connects to
            self.input_connect_list.append(proj.tolist())


    def set_neuronal_parameters(self, t_hyp_std = 0.0, V_hyp_0_std = 0.0):
        """
        sets up the connectivity matrix between network neurons
        """
        self.neuron_dicts = []
        self.all_weights = []
        # Here we define the neuronal parameters
        for i, pre_n in enumerate(self.liquid):   # go over every pre-synaptic liquid neuron
            t_hyp   = random.uniform( self.t_hyp - t_hyp_std, self.t_hyp + t_hyp_std )
            V_hyp_0 = random.uniform( self.V_hyp_0 - V_hyp_0_std, self.V_hyp_0 + V_hyp_0_std ) # remember this value < 0
            self.neuron_dicts.append({"E_L":self.E_L,
                                      "C_m":self.tau_m/self.r_inp,
                                      "tau_m":self.tau_m,
                                      "tau_syn_ex":self.tau_syn_ex,
                                      "tau_syn_in":self.tau_syn_in,
                                      "V_th":self.V_th,
                                      "t_ref":self.t_ref,
                                      "V_m":self.V_reset,
                                      "V_reset":self.V_reset,
                                      "I_e":self.I_e,
                                      "t_hyp":t_hyp,
                                      "V_hyp_0":V_hyp_0}) # C_m in pF
            if self.group_id[i] == 0.: # if this is an exc. neuron
               self.all_weights.append(self.exc_w)
            else:   # an inh. neuron
               self.all_weights.append(self.inh_w)


    def random_network_connect(self, exclude_self=True):
        """
        randomly connects network via random numbers
        """
        self.connection_list = []
        for i, pre_n in enumerate(self.liquid):
            p_con = []
            excluded = pre_n if exclude_self else -1  # -1 is not any neuron
            for i, post_n in enumerate(self.liquid):
                if np.random.rand() <= self.p_connect_ran and post_n != excluded:
                    p_con.append(post_n)
            self.connection_list.append(p_con)


    def set_experiment(self, voltmeters=False):
        """
        creates all the nest neurons, detectors and so on
        uses the parameters set in the __init__
        """
        print 'Setting up network and spike detectors'
        nest.sr("M_WARNING setverbosity")
        nest.ResetKernel()  # ResetKernel erases any existing network and restarts NEST
        nest.SetKernelStatus({"overwrite_files": True, "resolution": self.time_res})
        self.liquid = nest.Create("iaf_psc_exp_adp", self.n_total)    # create liquid
        nest.SetStatus(self.liquid, self.neuron_dicts)
        nest.SetDefaults('spike_generator',{'allow_offgrid_spikes':True}) # ,'precise_times':True
        if self.n_inputs:  # must initiate identities of the input trains
           self.inputs = nest.Create('spike_generator', self.n_inputs)
        self.connect_liquid_neurons()
        self.connect_spike_detectors()
        if voltmeters: self.connect_voltmeters() # memory-expensive


    def connect_liquid_neurons(self):
        for i, pre_synap_neuron in enumerate(self.liquid):   # connect the liquid neurons
            nest.Connect([pre_synap_neuron], self.connection_list[i],
                         syn_spec = {'weight': self.all_weights[i], 'delay':self.delay})


    def connect_spike_detectors(self):
        # connect the liquid neurons to spike detectors
        self.spike_detectors = nest.Create('spike_detector',len(self.liquid))
        nest.SetStatus(self.spike_detectors, {'to_file': False, 'to_screen': False, 'n_events': 0})
        for i, neuron in enumerate(self.liquid):
            nest.Connect(neuron, self.spike_detectors[i]) # connect liquid neurons to spike detectors


    def connect_voltmeters(self):
        ## the voltmeter/multimeter
        self.voltmeters = nest.Create("multimeter", len(self.liquid))
        nest.SetStatus(self.voltmeters, {"withgid": True, "to_file": False,
                                         "interval": self.time_res, "withtime": True,
                                         "record_from":["V_m", "V_hyp"]})
        for i in range(len(self.liquid)):
            nest.Connect(self.voltmeters[i], self.liquid[i])


    def connect_inputs_to_network(self, input_spike_trains=[], inp_weight=0.0, inp_delay=0.0):
        ## connects input trains to liquid
        for i, input_fiber in enumerate(self.inputs):
            nest.SetStatus([input_fiber], {'spike_times':input_spike_trains[i]})
            for j, neuron in enumerate(self.input_connect_list[i]):
                nest.Connect([input_fiber], [neuron],
                             syn_spec = {'weight': inp_weight, 'delay': inp_delay})


    def simulate(self, stimulus, sim_time=0.0, inp_weight=0.0, inp_delay=0.0,
                 voltmeters = False):
        """
        simulates a NN for sim_time ms
        """
        self.set_experiment(voltmeters=voltmeters)  # this resets/reconnects the network and spike detectors
        self.sim_time = stimulus.duration + 100.0 if sim_time==0.0 else sim_time
        self.stimulus = stimulus  # so that later it's possible to access by attribute
        assert len(stimulus.spike_trains) == len(self.inputs)
        self.connect_inputs_to_network(input_spike_trains = stimulus.spike_trains,
                                       inp_weight = inp_weight,
                                       inp_delay = inp_delay)
        nest.Simulate(self.sim_time)


    def spike_trains(self):
        spike_trains = []
        neurons = nest.GetStatus(self.spike_detectors, 'events')
        for i, nrn in enumerate(neurons):
            spike_trains.append(nrn.get('times'))
        return spike_trains


    def psth(self, sim_time, bin_size):
        """
        Returns 2D array with the liquid neurons spike psth.
        Note this is jus a plain psth, without considering gaps, etc.
        """
        psth = []
        bins = np.arange(sim_time/bin_size + 1) * bin_size
        bincenters = bins[0:-1] + ( bin_size / 2.0 )
        spike_trains = self.spike_trains()
        for i, spikes in enumerate(spike_trains):
            if len(spikes):
                t = np.asarray(np.histogram(spikes, bins)[0], dtype='float')
            else:
                t = np.zeros(len(bins)-1)
            psth.append(t)
        return bins, bincenters, np.array(psth)


    def read_voltage_file(self):
        """
        read voltmeters values from files..useless now
        """
        for i in range(len(self.voltmeters)):
            st = 'voltmeter-' + str(self.voltmeters[i])  + '-0.dat'
            print 'reading from file ', st
            a = np.loadtxt(st)
            self.volt_measured.append(a)


    def read_voltmeters(self):
        """
        reads multimeter values and saves the summed membrane potential.
        """
        self.potentials_measured = []
        potentials = nest.GetStatus(self.voltmeters, 'events')
        if potentials==[]:
           print 'sorry no potential measurements'
           return 0
        for i, dic in enumerate(potentials):
            v = dic.get("V_m") + dic.get("V_hyp", 0.0)
            self.potentials_measured.append(v)
        self.times = dic["times"]


    def plot_voltmeters(self, n_plots=20):
        """
        plots the first n_plots membrane potentials
        """
        plt.figure()
        self.read_voltmeters()   # read the voltmeters
        if self.potentials_measured==[]:
           print 'sorry no potential measurements to plot'
           return 0
        n_plots = min(n_plots, len(self.potentials_measured))
        for i in range(n_plots):    # plot every subplot
            nrn = i
            colour = 'b' if (nrn+1 in self.inh_group) else 'r'
            plt.subplot(n_plots, 1, i+1)
            v = self.potentials_measured[nrn]
            plt.plot(self.times, v, colour)
            plt.legend([nrn+1])
        plt.xlabel('time[ms]')
        plt.show()


if __name__=='__main__':
    neuronal_params = {'exc_w':10.0, 'inh_w':-10.0,
                       'syn_delay':1.0,
                       'tau_m':30.0, 'r_inp':10.0,
                       'tau_syn_ex':10.0, 'tau_syn_in':10.0,
                       't_hyp':10.0, 't_hyp_std':0.0,
                       'V_hyp_0':-10.0, 'V_hyp_0_std':0.0}
    network_params = {'n_total':2, 'n_inputs':1,
                     'inp_conn':1.0, 'p_exc':1.0,
                     'p_connect_ran':0.0}
    params = {}
    params.update(network_params)
    params.update(neuronal_params)
    lsm = LSM(**params)

    pattern_params = [[{'duration':50},{'duration':50},{'duration':50}]]
    gaps = [[{'duration':50},{'duration':50},{'duration':50}]]
    labels = [1,1,1]
    stimulus = Stimulus(pattern_params=pattern_params, gaps=gaps, labels=labels,
                        pattern_maker=poisson_spikes, description='simple stimulus test')
    #stimulus.spike_trains = [[10.0, 50.0, 90.0]]
    lsm.simulate(stimulus=stimulus, sim_time=400.0, inp_weight=10.0,
                 inp_delay=1.0, voltmeters=True)
    lsm.plot_voltmeters(n_plots=2)

    #bins, bincenters, psth = AnalytialTools.psth(lsm.sim_time, bin_size=10.0)
    #gap_psths = AnalyticalTools.multiple_bins_gap_psth(lsm.spike_trains(),
    #                                                   lsm.stimulus.gaps,
    #                                                   lsm.stimulus.pattern_params,
    #                                                   10.0,
    #                                                   6)
