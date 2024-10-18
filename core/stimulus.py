"""
This is a class written to produce objects that generate spatio-temporal patterns.
The patterns are imagined to consist of "bursts" and "gaps".
The class object takes in 3 arguments:
1.) pattern_params: A list of n lists.  n denotes the number of spike trains (fibers).
                    Each sublist is a list of m dictionaries,
                    where m is the number of patterns.
                    Each dictionary contains at least a "duration" (ms) key: {"duration":300, ....}
2.) gaps: Another list of n lists for the gap sizes along each fiber.
          Each sublist is a list of m gap size dicts (ms): {"duration":150}
3.) pattern_maker: A pattern generator that takes an element of the above sublist as input,
                   and outputs a pattern.
4.) labels: pattern labels.  Should be a list of numbers, of length = len(pattern_params[i]).

along each fiber, it'll look like gap/pattern/gap/pattern/gap/pattern/........
The object will also store every single pattern as a list of n lists,
identical to the input parameters.
"""
import numpy as np
from poisson_spikes import poisson_spikes


class Stimulus(object):


    def __init__(self, pattern_params, gaps, labels, pattern_maker, description, *args, **kwargs):
        self.pattern_params = pattern_params
        self.gaps = gaps
        self.labels = labels
        self.pattern_maker = pattern_maker  # this is a function that takes arguments
        self.description = description  # verbose description of what this stimulus is about
        #self.end_padding = end_padding  # in ms.  pads the end of stimulus with an empty gap

        if len(pattern_params) != len(gaps):
            raise Exception("Number of fibers do not match between pattern list and gap list")
        for i in range(len(pattern_params)): # This requirement may be relaxed if need be
            if len(pattern_params[i]) != len(pattern_params[0]):
                raise Exception("Number of patterns differ between fibers")
            if len(gaps[i]) != len(gaps[0]):
                raise Exception("Number of gaps differ between fibers")
            if len(pattern_params[i]) != len(gaps[i]):
                raise Exception("number of patterns and gaps along fiber %s do not match" % i)
        self.make_stimulus()


    def make_stimulus(self):
        """
        creates the stimulus, and stores all patterns
        """
        self.spike_trains = [] # initialize stimulus
        self.patterns = []
        self.duration = 0.0
        for i, param_list in enumerate(self.pattern_params): # iterate over each fiber
            fiber = [] # will be a list of spike times
            patts = [] # will be a list of sublists, each sublist a pattern
            time = 0.0  # start from time = 0
            for j, params in enumerate(param_list): # iterate over the patterns for each fiber
                time += self.gaps[i][j]['duration'] # move time forward
                p = self.pattern_maker(**params) # spike times, anchored at time = 0
                patts.append( p )       # store the pattern
                spike_times = [q + time for q in p]  # the pattern in real time
                fiber += spike_times    # insert the spike times into this fiber
                time  += params["duration"] # move time forward
            self.duration = time if time > self.duration else self.duration
            self.spike_trains.append(fiber)
            self.patterns.append(patts)
        #self.duration += self.end_padding


    def noisy_stimulus(self, noise_rate=5.0):
        """
        a method to create a noisy version of the clean stimulus,
        by introducing Poissinon spikes of rate = noise_rate along each fiber.
        Since noise is by definition non-pattern, we do not save it as self.noisy_stimulus,
        but merely return it to the caller.
        adds an extra time padding to the back-end, with a span of gaps[-1][-1]
        """
        noisy_spike_trains = []
        for fiber in self.spike_trains:
            noise = poisson_spikes(duration = self.duration,
                                   rate = noise_rate).tolist()
            noisy_spike_trains.append( sorted(fiber + noise) )
        return noisy_spike_trains


    def jitter(self, jittertime=0.0):
        """
        jitters the stimulus according to a normal distribution, with std = jittertime
        """
        jittered_spike_trains = []
        for fiber in self.spike_trains:
            jitter = np.random.normal(0.0, jittertime, len(fiber))
            jittered_spike_trains.append( (np.array(fiber) + jitter).tolist() )
        return jittered_spike_trains


    def psth(self, duration, time_bin):
        psth = []
        bins = np.arange(duration/time_bin+1)*time_bin
        bincenters = bins[0:-1] + ( time_bin / 2.0 )
        for i, spikes in enumerate(self.spike_trains): # enumerate over each fiber
            if len(spikes):
                t = np.asarray(np.histogram(spikes, bins)[0], dtype='float')
            else:
                t = np.zeros(len(bins)-1)
            psth.append(t)
        return bins, bincenters, np.array(psth)


if __name__ == "__main__":
    pattern_params = [[{'duration':100},{'duration':200},{'duration':300}],
                      [{'duration':200},{'duration':200},{'duration':200}]]
    labels = [1,2,3]
    gaps = [[{'duration':100},{'duration':100},{'duration':100}],
            [{'duration':200},{'duration':200},{'duration':200}]]
    stimulus = Stimulus(pattern_params = pattern_params,
                        gaps = gaps,
                        labels = labels,
                        pattern_maker = poisson_spikes,
                        description = 'simple stimulus test')
