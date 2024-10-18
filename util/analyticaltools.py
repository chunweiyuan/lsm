"""
Here reside the routines used to analyze network outputs
"""
from __future__ import division
import numpy as np
from sklearn.decomposition import PCA
import copy


class AnalyticalTools(object):

    @classmethod
    def mutual_info(cls, labels, psth):
        """
        Given the labels (1-by-p) and the psth (m-by-p), computes the mutual information between
        each row (neuron) and the labels
        Taking H(X) = - sum_X p(X) ln p(X)
        and  H(X|Y) = - sum_Y p(Y) sum_X p(X|Y) ln p(X|Y)
        Mutual information is computed as I = H(X) - H(X|Y),
        where X is a row from psth, and Y are the labels.
        """
        assert len(labels) == len(psth[0])
        I = np.zeros(len(psth)) # i of I is the entropy of the ith row
        classes = set(labels)
        for i, row in enumerate(np.array(psth)): #enumerate over each row
            I[i] = cls.entropy(row) - cls.conditional_entropy(labels, row)
        return I


    @staticmethod
    def entropy(obs):
        # obs is a list or a 1-D numpy.array of integers (observations)
        xs = np.array(obs) # this ensures a deep copy
        h = 0.0
        for x in set(xs):
            px = float(len( xs[xs==x] )) / float( len(xs) ) # P(X=x)
            h += (-1) * px * np.log2( px )
        return h


    @classmethod
    def conditional_entropy(cls, labels, obs):
        assert len(labels) == len(obs)  # must be true!
        xs = np.array(obs)
        ys = np.array(labels)
        h = 0.0
        for y in set(labels):
            py = float(len( ys[ys==y] )) / float( len(ys) )
            h += py * cls.entropy( xs[ys==y] ) # no negative sign here.  that's already in entropy.
        return h


    @classmethod
    def binary_mutual_info(cls, labels, psth, n=1):
        # computes the binary (zero vs. non-zero) version of mutual_info
        m = np.array(psth)
        m[m>0] = 1
        return cls.mutual_info(labels, m)


    @staticmethod
    def pca(x, n_components=None, copy=True, whiten=False):
        """
        Uses the scikit learn pacakge here.
        Because my data matrix x has each row being a dimension, and each column an observation,
        which is the reverse of what the sklearn package does, I must tranpose my x matrix to fit.
        """
        bot = PCA(n_components=n_components, copy=copy, whiten=whiten)
        bot.fit(np.array( np.transpose(x) ))  # must transpose my matrix
        # bear in mind that the outputs will have the components as rows, not columns.
        return bot


    @staticmethod
    def pc_richness(eigenvalues, threshold=0.9):
        """
        This is a routine designed to find out the number of dimensions needed to reach 90%
        of the total variance.  The implementation is fairly simple,
        since the eigenvalues are the explained variances associated with the eigenvectors,
        and are sorted from max to min.
        That means all that needs to be done is first summing over the eigenvalues,
        and then determining the minimum number of eigenvalues needed to reach 90% of the total.
        """
        vsum = np.sum(eigenvalues)
        percentage = 0.0
        for i in range(len(eigenvalues)):
            percentage = percentage + ( eigenvalues[i] / vsum )
            if percentage >= threshold: break
        # "count" gives the number of eigenvalues counted.  Of course, since "i" starts at 0, count = i + 1
        count = i + 1
        return count


    @staticmethod
    def pc_transients(eigenvectors, psth):
        """
        This is a function designed to take in the eigenvectors of the PCA result,
        and spit out the prominence of each principal component along the time axis.
        The prominence of a particular principal eigenvector is simply determined
        by its projection (dot product) with the mean-corrected network state vector at a given time

        psth gives the original data, which should be and M-by-N matrix, with M = dimensions and N = time bins
        The principal components are the column vectors of the M-by-M "eigenvectors".

        The output should be a M-by-N matrix, where M are the neurons and N is the number of time bins.
        The easiest way to obtain the output is to take the transpose of "eigenvectors", and then
        calculate the matrix product ( eigenvectors^T * psth )

        The columns of "transients" are not normalized
        """
        n = len(psth[0,:]) # first need to know the number of trials
        m = psth - np.mean(psth,1).reshape(-1,1)  # construct the zero-mean rows
        # transients is essentially a transformed matrix.
        # The transformation goes like T = PM, where M is the zero-meaned original data set
        # P is the transpose of the eigenvectors put together.
        # This means that the jth column of T gives the projections of the jth column of M on the eigenvectors.
        transients = np.dot(np.transpose(eigenvectors),m)
        return transients


    @staticmethod
    def plot_patterns(patterns, xaxis):
        """
        This is just a function to plot multiple data curves over time.
        "patterns" contains the M data curves in its M rows.  Each curves has N data points.
         xaxis" is a N-by-1 row vector containing the x-coordinate values at which the data points are taken.
        """
        import matplotlib
        matplotlib.use("MacOSX")
        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(xaxis,np.transpose(patterns))
        matplotlib.pyplot.xlabel('time')
        matplotlib.pyplot.ylabel('activity')


    @staticmethod
    def pds(psth):
        """
        This routine takes in the time-binned spike sums and produces the power density spectrum (PDS),
        by taking the product of the fft and its cojugate
        I use the rfft numpy built-in function here, which only computes up to the nyquist frequency
        """
        fourier_coeffs = np.fft.rfft(psth)
        pds = np.abs(fourier_coeffs)**2.0
        pds = pds / np.max(pds)
        return pds


    @staticmethod
    def pds_smoothed(psth, std, width):
        fourier_coeffs = np.fft.rfft(spike_sums)
        pds = np.abs(fourier_coeffs)**2.0
        # now construct my smoothing kernel to be used for the convolution.
        # A Gaussian of peak value 1.0 and std=2.0
        smoothing_kernel = np.zeros(int(2 * width + 1))
        for i in range(len(smoothing_kernel)):
            smoothing_kernel[i] = Gaussian(np.float(i-width),0.0,std)
        pds_smoothed = np.convolve(pds,smoothing_kernel) # this is the convolution part
        pds_smoothed = pds_smoothed / np.max(pds_smoothed)
        return pds_smoothed


    @staticmethod
    def gap_psth(spike_trains, gaps, pattern_params, bin_size, time_shift):
        """
        psth relative to gap-burst interfaces.
        spike_trains:  array of n spike trains, with n being the number of dimensions.
        gaps:  Stimulus.gaps (ex. [[{duration: 100}, {duration:200}], [{duration:50}, {duration:80}]] )
        patterns:  Stimulus.pattern_params  (similar to above)
        assumes the stimulus follows the gap/burst/gap/burst/gap/burst ... order.
        """
        psth =[]
        for i, fiber in enumerate(spike_trains): # enumerate over the input dimensions
            hist = [] # hist might be a misnomer.  This really is just a list of spike counts.
            gap_end = 0.0
            for j, gap in enumerate(gaps[0]):  # only look at gap params of the 1st input fiber
                gap_end += gap['duration']
                tbin = [gap_end + time_shift, gap_end + time_shift + bin_size - 0.01] # -0.01 to not include both edges
                count = np.histogram(fiber, tbin)[0]  # this extracts the spike count of this dimension at this bin
                hist.append(count[0])
                gap_end += pattern_params[0][j]['duration'] # pattern durations of 1st input fiber
            hist = np.asarray(hist, dtype='float')
            psth.append(hist)
        psth = np.array(psth) # psth is now n-by-m, where n is the number of fibers, m is number of samples.
        return psth


    @classmethod
    def multiple_bins_gap_psth(cls, spike_trains, gaps, pattern_params, bin_size, number_of_bins):
        """
        Computes psth, with time shift that is i * bin size,
        where is ranges from 0 to number_of_bins.
        Collects all resultant psth, and concatenates them along dimension 1.
        """
        for i in range(number_of_bins):
            psth = cls.gap_psth(spike_trains, gaps, pattern_params, bin_size, i*bin_size)
            # each psth is n-by-m, where n is the number of fibers, m is the number of samples.
            if i == 0:
                all_psth = psth
            else:
                all_psth = np.concatenate((all_psth, psth), 1)
        return all_psth
