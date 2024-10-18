"""
This is a class written to produce stimuli with anomalies.
The simplest example is periodic pure tone bursts with
occasional anomalies that are either higher pitch or longer duration.
This class in essence generates the inputs to the Stimulus class object,
and then returns a Stimulus object, which contains all the labels, etc.
"""
from stimulus import Stimulus
from txtsets import TXTSets
from poisson_spikes import poisson_spikes


class OddBall(object):

    @classmethod
    def periodic_bursts(cls, burst_duration, burst_freq, gap_duration,
                        anomaly_duration, anomaly_freq, anomaly_fraction,
                        description, n_patterns=100, n_fibers=10,
                        randomize=False):
        """
        This method constructs a stimulus with repeated bursts of the same
        frequency and duration, interjected by anomalous bursts.
        The gaps are maintained the same.
        If randomize = False, then the anomalies are injected periodically.
        If randomize = True, then a periodic version is first generated,
        and then the whole thing is shuffled.
        """
        pattern_params = []
        gaps = []
        labels = []
        periodicity = int(1.0 / anomaly_fraction) # 1 in every n is anomalous
        for i in range(n_fibers):
            pattern_params.append([])
            gaps.append([])
            for j in range(n_patterns):
                if j > 0 and (j+1) % periodicity == 0:
                    # this burst is supposed to be the oddball
                    duration, rate = (anomaly_duration, anomaly_freq)
                    if i==0: labels.append(1)
                else: # normal
                    duration, rate = (burst_duration, burst_freq)
                    if i==0: labels.append(0)
                pattern_params[i].append({'duration':duration, 'rate':rate})
                gaps[i].append({'duration':gap_duration})
        stimulus = Stimulus(pattern_params=pattern_params,
                            gaps=gaps,
                            labels=labels,
                            pattern_maker=poisson_spikes,
                            description=description)
        if randomize is True:
            stimulus = TXTSets.shuffle(stimulus)
        return stimulus


if __name__ == "__main__":
    stimulus = OddBall.periodic_bursts(burst_duration=100., burst_freq=100.,
                                       gap_duration = 100.0,
                                       anomaly_duration=50, anomaly_freq=1000.,
                                       anomaly_fraction=0.2, description='simple test',
                                       n_patterns=10, n_fibers=2, randomize=True)
