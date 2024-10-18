import cPickle as pickle
import os

from core.lsm import LSM
from core.txtsets import TXTSets
from core.oddball import OddBall
from core.svmlinear import SVMLinear
from util.analyticaltools import AnalyticalTools
from config import (NEURONAL_PARAMS, NETWORK_PARAMS, STIMULUS_PARAMS,
                    TXTSET_PARAMS, SIMULATION_PARAMS, INPUT_PARAMS)


class SimulationManager(object):
    """
    A simulator object should contain all the necessary objects
    to run a lsm simulation pipeline:
    1.) create lsm and stimulus objects
    2.) train, x-validate, and test classifier using network outputs
    3.) collect statistics and perform analysis
    4.) export analytical results to files
    """

    def __init__(self, neuronal_params, network_params, stimulus_params,
                 txt_params, inp_params, sim_params):
        """
        network is the lsm object
        stimulus is the stimulus object
        reps is the number of times we run the simulation.
        """
        self.neuronal_params = neuronal_params
        self.network_params = network_params
        self.stimulus_params = stimulus_params
        self.txt_params = txt_params  # only used by self.make_txtsets()
        self.inp_params = inp_params
        self.sim_params = sim_params  # only used by self.run()

        # now set up the neural network (kept fixed during different runs)
        params = {}
        params.update(network_params) # use update, so we don't change object.
        params.update(neuronal_params)
        self.network = LSM(**params) # params define the entire neural network
        #self.network.connect_voltmeters()  # WARNING: memory-expensive!

    def make_txtsets(self, stimulus):
        """
        Makes the training, cross-validation, and test sets.

        Args:
            stimulus (Stimulus): a Stimulus object defined in stimulus.py.

        NOTE: assues self.stimulus is already defined
        """
        self.txtsets = TXTSets(stimulus, n=self.txt_params['pattern_reps'])

    def set_svmlin(self, training_set, labels):
        """
        Given training set and labels, trains/cross-validates the linear svm
        classifier to obtain the optimal fit.
        Internally, the cross-validation is done on a grid search
        using the training data to arrive at the best hyperparameters.
        """
        self.svmlin = SVMLinear(training_set, labels) # sets hyperparameters.
        self.svmlin.fit(training_set, labels)  # trains classifier.

    def save_pickle(self, filename):
        # save the workspace into a pickle file, with the proper name.
        pickle.dump(self.__dict__, open(filename, "wb"))

    @classmethod
    def read_pickle(cls, filename):
        data = pickle.load(open(filename, "rb"))
        return data

    def simulate_network(self, stimulus):
        """
        Runs simulation with given stimulus,
        and collect psth based on given bin_size and time_shift.
        NOTE: the returned psth is a gap psth,
              charaterized by its bin size and time shift from burst onset.
        """
        self.network.simulate(stimulus=stimulus,
                              sim_time=self.sim_params['sim_time'],
                              inp_weight=self.inp_params['inp_w'],
                              inp_delay=self.inp_params['inp_delay'],
                              voltmeters=True)

    def train_and_test_input(self, bin_size, time_shift):
        """
        Trains classifier using the training set,
        returns training score and test score.
        """
        psth = AnalyticalTools.gap_psth(
                   spike_trains=self.txtsets.training_set.spike_trains,
                   gaps=self.txtsets.training_set.gaps,
                   pattern_params=self.txtsets.training_set.pattern_params,
                   bin_size=bin_size,
                   time_shift=time_shift )
        self.set_svmlin(psth.T, self.txtsets.training_set.labels)  # trains
        training_score = self.svmlin.score(psth.T, self.txtsets.training_set.labels)

        psth = AnalyticalTools.gap_psth(
                   spike_trains=self.txtsets.test_set.spike_trains,
                   gaps=self.txtsets.test_set.gaps,
                   pattern_params=self.txtsets.test_set.pattern_params,
                   bin_size=bin_size,
                   time_shift=time_shift )
        test_score = self.svmlin.score(psth.T, self.txtsets.test_set.labels)

        return training_score, test_score

    def train_and_test_network(self, bin_size, time_shift):
        """
        Runs training stimulus and test stimulus through the network
        to collect transformed results.
        Use said results to train and test classifier.
        """
        self.simulate_network(stimulus=self.txtsets.training_set)
        psth = AnalyticalTools.gap_psth(
                   spike_trains=self.network.spike_trains(),
                   gaps=self.txtsets.training_set.gaps,
                   pattern_params=self.txtsets.training_set.pattern_params,
                   bin_size=bin_size,
                   time_shift=time_shift)
        self.set_svmlin(psth.T, self.txtsets.training_set.labels)
        training_score = self.svmlin.score(psth.T,
                                           self.txtsets.training_set.labels)

        self.simulate_network(stimulus=self.txtsets.test_set)
        psth = AnalyticalTools.gap_psth(
                   spike_trains=self.network.spike_trains(),
                   gaps=self.txtsets.test_set.gaps,
                   pattern_params=self.txtsets.test_set.pattern_params,
                   bin_size=bin_size,
                   time_shift=time_shift)
        test_score = self.svmlin.score(psth.T, self.txtsets.test_set.labels)

        return training_score, test_score

    def run(self, out_path='', bin_size=30.0, time_shift=0.0):
        """
        This is a routine written to run a batch of simulations.
        In this version, all parameters are set within the code.  Hence no input parameters.
        NOTE: currently this is tailored to running gap simulations,
              but if one sets the gap sizes to be 0 then it's straight-forward
              to translate to a continuous stimulus.
        NOTE: the odd-ball stimulus usage is hard-coded as of now.
        """
        if not os.path.exists(out_path):
            out_path = os.getcwd() + '/out'
        import pdb; pdb.set_trace()
        for i in range(self.sim_params['number_of_runs']):
            # a new stimulus for each run
            stimulus = OddBall.periodic_bursts(**self.stimulus_params)
            self.make_txtsets(stimulus) # create the training/x-validation/test sets.
            # train and test on un-transformed input data
            input_training_score, input_test_score =\
                self.train_and_test_input(bin_size, time_shift)
            # now train on network-transformed input data
            network_training_score, network_test_score =\
                self.train_and_test_network(bin_size, time_shift)
            with open(out_path + '/data.txt', 'a') as outfile:
                outfile.write('{},{},{},{}\n'.format(input_training_score,
                                                     input_test_score,
                                                     network_training_score,
                                                     network_test_score))


def main(neuronal_params, network_params, stimulus_params, txt_params,
         sim_params, inp_params):
    """
    This is function that contains all the input keyword arguments.
    Creates the network and stimulus objects,
    and feeds them into the simulation manager.
    """
    sim_manager = SimulationManager(neuronal_params=neuronal_params,
                                    network_params=network_params,
                                    stimulus_params=stimulus_params,
                                    txt_params=txt_params,
                                    inp_params=inp_params,
                                    sim_params=sim_params)
    sim_manager.run(bin_size=30.0, time_shift=0.0)


if __name__=='__main__':
   main(neuronal_params=NEURONAL_PARAMS,
        network_params=NETWORK_PARAMS,
        stimulus_params=STIMULUS_PARAMS,
        txt_params=TXTSET_PARAMS,
        sim_params=SIMULATION_PARAMS,
        inp_params=INPUT_PARAMS)
