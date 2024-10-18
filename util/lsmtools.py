import nest  # assume nest is installed
import matplotlib
matplotlib.use("MacOSX")  # MacOSX
import random
import numpy
import nest.voltage_trace
import nest.raster_plot
import sys
import pickle
import numpy.linalg
from poisson_spikes import poisson_spikes, multiple_trains

#TODO: Much of the following needs to be refactored., perhaps into analyticaltools.py.

def autocorrelation_collapsed(liquid, sim_time=500.0,time_bin=1.0):

    """this routine counts the number of total spikes in each bin, and returns the result in a 1-D vector
       then it sets all non-zero elements to 1, and then computes the autocorrelation of the vector.
    """

    spike_times = nest.GetStatus(liquid.spike_detectors, 'events')
    psth = []
    ntotal = len(spike_times) # this should give the number of neurons

    bins = numpy.arange(sim_time/time_bin+1)*time_bin # the elements here "bracket" the bins
    bincenters = bins[0:-1] + ( time_bin / 2.0 )

    for nrn in range(ntotal):  # enumerate over every neuron (dimension)
        spikes = spike_times[nrn].get("times") # get all the spikes from this particular neuron
        t = numpy.zeros(len(bins)-1) # the actual number of bins
        if len(spikes):
           #t = numpy.histogram(spikes, bins, new=True)[0]
           t = numpy.histogram(spikes, bins)[0]
        psth.append(t)

    psth = numpy.array(psth)

    #psth[psth>0] = 1.0

    spike_sums = numpy.sum(psth,axis=0)

    #spike_sums[spike_sums>0] = 1 # this sets all the non-zero elements t o1

    spike_sums = spike_sums[100:] # I think the first 100 probably don't matter
    # now we need to compute the autocorrelation of the resultant vector
    autocorr = numpy.correlate(spike_sums,spike_sums,'full')

    return autocorr



def coefficient_of_variation(liquid):
    """
       This is a function that computes the coefficient of variation (CV) of the interspike intervals of the whole network.  The routine first computes the CV of individual neurons, then averages the CVs of the N network excitatory neurons.
    """

    exc_network_spikes = nest.GetStatus(liquid.spike_detector,'events')
    exc_senders = exc_network_spikes[0]['senders'] # this is a numpy.ndarray
    exc_times = exc_network_spikes[0]['times'] # this is a numpy.ndarray

    sorted_indices = numpy.argsort(exc_senders) # sort them by neuron index
    exc_senders = exc_senders[sorted_indices]
    exc_times = exc_times[sorted_indices]
    exc_senders = exc_senders.tolist() # convert into list
    exc_times = exc_times.tolist()
    #print exc_senders
    #print exc_times
    ISI_avgs = numpy.zeros(len(liquid.liquid)) # a numpy vector of the mean ISI of each neuron.  If the value is zero, that means this neuron never fired
    CV_avgs = numpy.zeros(len(liquid.liquid)) # a numpy vector of the CV of each neuron.  If the value is zero, that means this neuron never fired

    for i in range(len(liquid.liquid)):
        if liquid.liquid[i] in exc_senders:
           j = exc_senders.index(liquid.liquid[i]) # this gives the index to the first appearance of liquid.liquid[i]
           n = exc_senders.count(liquid.liquid[i]) # this counts the number of times the neuron liquid.liquid[i] fires
           spike_times = exc_times[j:j+n] # a numpy vector of spike times of neuron liquid.liquid[i]

           spike_times = numpy.array(spike_times) # convert the list into numpy.array for vector manipulation
           spike_times = numpy.sort(spike_times) # sort the times

           ISIs = spike_times[1:n] - spike_times[0:n-1] # these are the firing intervals of this neuron

           ISI_avg = numpy.mean(ISIs) # this is the average of the firing intervals of this neuron
           CV = numpy.std(ISIs) / ISI_avg # this is the std of this neuron's firing intervals, divided by the mean

           ISI_avgs[i] = ISI_avg
           CV_avgs[i] = CV

     # now let's compute the means of the ones that actually fire
    list = ISI_avgs > 0

    CV = numpy.mean(CV_avgs[list])

    #return CV, ISI_avgs, CV_avgs
    return CV



def count_spikes(liquid, sim_time=500.0,time_bin=25.):
    """this routine counts the number of total spikes in each bin, and returns the result in a 1-D vector"""
    spike_times = nest.GetStatus(liquid.spike_detectors, 'events')
    psth = []
    ntotal = len(spike_times) # this should give the number of neurons
    bins = numpy.arange(sim_time/time_bin+1)*time_bin # the elements here "bracket" the bins
    bincenters = bins[0:-1] + ( time_bin / 2.0 )
    for nrn in range(ntotal):  # enumerate over every neuron (dimension)
        spikes = spike_times[nrn].get("times") # get all the spikes from this particular neuron
        t = numpy.zeros(len(bins)-1) # the actual number of bins
        if len(spikes):
           #t = numpy.histogram(spikes, bins, new=True)[0]
           t = numpy.histogram(spikes, bins)[0]
        psth.append(t)
    psth = numpy.array(psth)
    #psth[psth>0] = 1.0
    spike_sums = numpy.sum(psth,axis=0)
    return spike_sums



def count_spikes_smoothed(liquid, sim_time, time_bin, std, width):

    """this routine counts the number of total spikes in each bin, and returns the result in a 1-D vector.
       the spikes of each neuron are binned, then the 1-D vector of 1's and 0's is covolved with a smoothing kernel.  The bin size here should be small enough such that only 1's and 0's exist, but it's by no means necessary.  The choice of smoothing kernel here is a Gaussian kernel.
    """
    spike_times = nest.GetStatus(liquid.spike_detectors, 'events')
    psth = []
    ntotal = len(spike_times) # this should give the number of neurons

    bins = numpy.arange(sim_time/time_bin+1)*time_bin # the elements here "bracket" the bins
    bincenters = bins[0:-1] + ( time_bin / 2.0 )

    # now construct my smoothing kernel to be used for the convolution.  A Gaussian of peak value 1.0 and std=2.0
    smoothing_kernel = numpy.zeros(int(2 * width + 1))
    for i in range(len(smoothing_kernel)):
        smoothing_kernel[i] = gaussian(numpy.float(i-width),0.0,std)
    #smoothing_kernel = numpy.array([0.135,0.278,0.487,0.726,0.923,1.0,0.923,0.726,0.487,0.278,0.135])
    #print smoothing_kernel

    for nrn in range(ntotal):  # enumerate over every neuron (dimension)
        spikes = spike_times[nrn].get("times") # get all the spikes from this particular neuron
        t = numpy.zeros(len(bins)-1) # the actual number of bins
        if len(spikes):
           #t = numpy.histogram(spikes, bins, new=True)[0]
           t = numpy.histogram(spikes, bins)[0]

        t = numpy.convolve(t,smoothing_kernel) # this is the convolution part
        psth.append(t)

    psth = numpy.array(psth)

    #psth[psth>0] = 1.0

    spike_sums = numpy.sum(psth,axis=0)

    return spike_sums



def count_bipopulation_spikes_smoothed(liquid, sim_time, time_bin, std, width):

    """this routine counts the number of total spikes in each bin, and returns the result in a 1-D vector.
       the spikes of each neuron are binned, then the 1-D vector of 1's and 0's is covolved with a smoothing kernel.  The bin size here should be small enough such that only 1's and 0's exist, but it's by no means necessary.  The choice of smoothing kernel here is a Gaussian kernel.
    """
    spike_times = nest.GetStatus(liquid.spike_detectors, 'events')
    ntotal = len(spike_times) # this should give the number of neurons

    bins = numpy.arange(sim_time/time_bin+1)*time_bin # the elements here "bracket" the bins
    bincenters = bins[0:-1] + ( time_bin / 2.0 )

    # now construct my smoothing kernel to be used for the convolution.  A Gaussian of peak value 1.0 and std=2.0
    smoothing_kernel = numpy.zeros(int(2 * width + 1))
    for i in range(len(smoothing_kernel)):
        smoothing_kernel[i] = Gaussian(numpy.float(i-width),0.0,std)
    #smoothing_kernel = numpy.array([0.135,0.278,0.487,0.726,0.923,1.0,0.923,0.726,0.487,0.278,0.135])
    #print smoothing_kernel

    population_total = int( ntotal / 2 )

    # Now the first population
    psth = []
    spike_num1 = []
    for nrn in range(population_total):  # enumerate over every neuron (dimension)
        spikes = spike_times[nrn].get("times") # get all the spikes from this particular neuron
        t = numpy.zeros(len(bins)-1) # the actual number of bins
        if len(spikes):
           #t = numpy.histogram(spikes, bins, new=True)[0]
           t = numpy.histogram(spikes, bins)[0]
           spike_num1.append(len(spikes)) # if this neuron spikes, take it into accout for average spike rate
        t = numpy.convolve(t,smoothing_kernel) # this is the convolution part
        psth.append(t)

    psth = numpy.array(psth)
    spike_sums = numpy.sum(psth,axis=0)
    spike_rate1 = numpy.mean(spike_num1) / sim_time * 1000.0 # spike rate in Hz.  The 1000.0 accounts for ms.
    # Now the 2nd population
    psth2 = []
    spike_num2 = []
    for nrn in range(population_total):  # enumerate over every neuron (dimension)
        spikes = spike_times[population_total + nrn].get("times") # get all the spikes from this particular neuron
        t = numpy.zeros(len(bins)-1) # the actual number of bins
        if len(spikes):
           #t = numpy.histogram(spikes, bins, new=True)[0]
           t = numpy.histogram(spikes, bins)[0]
           spike_num2.append(len(spikes))
        t = numpy.convolve(t,smoothing_kernel) # this is the convolution part
        psth2.append(t)
    psth2 = numpy.array(psth2)
    spike_sums2 = numpy.sum(psth2,axis=0)
    spike_rate2 = numpy.mean(spike_num2) / sim_time * 1000.0
    return spike_sums, spike_sums2, spike_rate1, spike_rate2



def isi_statistics(liquid, sim_time, time_bin, gap_endpoints, time_shift):
    """
       This is a routine that computes the ISI statistics of the network's response
     """
    exc_network_spikes = nest.GetStatus(liquid.spike_detector,'events') # this can be modified to looked at only exc neurons, or the whole network
    exc_senders = exc_network_spikes[0]['senders'] # this is a numpy.ndarray
    exc_times = exc_network_spikes[0]['times'] # this is a numpy.ndarray

    sorted_indices = numpy.argsort(exc_senders) # sort them by neuron index
    exc_senders = exc_senders[sorted_indices]
    exc_times = exc_times[sorted_indices]
    exc_senders = exc_senders.tolist() # convert to list
    exc_times = exc_times.tolist() # convert to list

    isi_avgs = numpy.zeros(len(liquid.liquid)) # a numpy vector of the mean ISI of each neuron.  If the value is zero, that means this neuron never fired
    cv_avgs = numpy.zeros(len(liquid.liquid)) # a numpy vector of the CV of each neuron.  If the value is zero, that means this neuron never fired
    all_isis = []

    #bin_edges = numpy.arange(sim_time/time_bin+1)*time_bin # these are actually bin edges

    for i in range(len(liquid.liquid)):

        isis_i = [] # this is eventually gonna become a list of all ISIs of this particular neuron

        if liquid.liquid[i] in exc_senders:

           j = exc_senders.index(liquid.liquid[i]) # this gives the index to the first appearance of liquid.liquid[i]
           n = exc_senders.count(liquid.liquid[i]) # this counts the number of times the neuron liquid.liquid[i] fires
           all_spike_times = numpy.sort(numpy.array(exc_times[j:j+n])) # a numpy vector of all spike times of neuron liquid.liquid[i]

           # now I must traverse over the desired bins and compute the individual ISI's in each bin
           for gap_endpoint in gap_endpoints: # this gives the number of patterns we need to analyze

               left_edge = gap_endpoint + time_shift
               right_edge = gap_endpoint + time_shift + time_bin

               spike_times = all_spike_times[(all_spike_times>=left_edge)*(all_spike_times<right_edge)]

               if len(spike_times) > 1: # it only makes sense to compute intervals if there are multiple spikes
                  isis = spike_times[1:len(spike_times)] - spike_times[0:len(spike_times)-1]

                  for n in range(len(isis)):
                      isis_i.append(isis[n])
                      all_isis.append(isis[n]) # tabulate all ISIs over all neurons over all times

        isis_i = numpy.array(isis_i)
        if len(isis_i) > 0:
           isi_avgs[i] = numpy.mean(isis_i)
        else:
           isi_avgs[i] = 0  # !!! this is important: if there are no spikes, the ISI avg is set to be 0

        if isi_avgs[i] > 0.: # if this neuron never fires, or fires only once in each bin, it's CV is 0
           cv_avgs[i] = numpy.std(isis_i) / isi_avgs[i]

    return isi_avgs, cv_avgs, all_isis



def multiple_fixed_input(ninput, input_spike_times):
    """
       This is where I construct multiple artificial input trains, with spike times defined here.
    """

    # now define the actual spikes in NEST
    inputs = nest.Create('spike_generator',ninput)

    # now insert the spike times
    for i in range(len(inputs)):
        nest.SetStatus([inputs[i]],{'spike_times':numpy.array(input_spike_times[i])})

    input_spike_detector = nest.Create('spike_detector')
    nest.SetStatus(input_spike_detector, {'to_file':False, 'to_screen':False})
    nest.ConvergentConnect(inputs, input_spike_detector)

    return input_spike_times, inputs, input_spike_detector



def plot_raster(liquid,input_spike_times,input_connect_list,plot_inputs=1, plot_outputs=1):
    "this little routine is tailored to plot the inputs spikes and network outputs on the same raster plot"
    matplotlib.pyplot.figure()

    if plot_inputs:
       [input_times,input_senders] = Input_Scatter(input_spike_times,input_connect_list)
       input_dots = matplotlib.pyplot.plot(input_times,input_senders)
       matplotlib.pyplot.setp(input_dots,marker='o',mfc='Orange',ms=4,mew=0.1,linewidth=0.0)

    if plot_outputs:
       inh_network_spikes1 = nest.GetStatus(liquid.group1_spike_detector,'events')
       inh_senders1 = inh_network_spikes1[0]['senders']
       inh_times1 = inh_network_spikes1[0]['times']
       inh_network_dots1 = matplotlib.pyplot.plot(inh_times1,inh_senders1)
       matplotlib.pyplot.setp(inh_network_dots1,marker='o',mfc='Purple',ms=4,mew=0.1,linewidth=0.0)
       #matplotlib.pyplot.setp(inh_network_dots1,marker='o',mfc='ForestGreen',ms=4,mew=0.1,linewidth=0.0)

       inh_network_spikes2 = nest.GetStatus(liquid.group2_spike_detector,'events')
       inh_senders2 = inh_network_spikes2[0]['senders']
       inh_times2 = inh_network_spikes2[0]['times']
       inh_network_dots2 = matplotlib.pyplot.plot(inh_times2,inh_senders2)
       #matplotlib.pyplot.setp(inh_network_dots2,marker='o',mfc='ForestGreen',ms=4,mew=0.1,linewidth=0.0)
       matplotlib.pyplot.setp(inh_network_dots2,marker='o',mfc='Purple',ms=4,mew=0.1,linewidth=0.0)

    matplotlib.pyplot.show()



def plot_voltmeters(voltmeters,n_plots=1,time_res=0.1):
    """
       This plots voltmeters
    """
    potentials_measured = []
    membrane_potentials = nest.GetStatus(voltmeters,'events')

    if membrane_potentials==[]:
           print 'sorry no potential measurements'
           return 0

    for i in range(len(membrane_potentials)):
            v = membrane_potentials[i].get("potentials")
            potentials_measured.append(v)

    matplotlib.pyplot.figure()

    n_plots = min(n_plots, len(potentials_measured))
    s_len = len(potentials_measured[0])
    time = numpy.arange(s_len)*time_res

    for i in range(n_plots):    # plot every subplot
        colour = 'b'
        nrn = i
        #if (nrn+1 in self.inh_group1): colour = 'r' # if inhibitory-red plot
        matplotlib.pyplot.subplot(n_plots,1,i+1)
        v = potentials_measured[nrn]
        matplotlib.pyplot.plot(time,v,colour)
        matplotlib.pyplot.legend([nrn+1])
    matplotlib.pyplot.xlabel('time[ms]')



def random_input_connect_exact(inputs,liquid,connectivity,weight,delay):
    """
       This connects the input spike trains to the liquid neurons.
       Assumes constant weight and delay
       The connections are random, but the NUMBER of connections is exact.
    """
    ntotal = liquid.n_total

    connect_number = int( numpy.float( ntotal ) * connectivity ) # exact number to connect to

    input_connect_list = []
    #print "inputs", inputs
    for i in range(len(inputs)):

        M = 0
        connect_list = []

        while (M != connect_number):
              slice = numpy.random.rand(ntotal)
              #slice = numpy.arange(ntotal) / numpy.float(ntotal)
              #numpy.random.shuffle(slice)
              boo0 = slice > connectivity
              boo1 = slice <= connectivity
              slice[boo0] = 0
              slice[boo1] = 1
              M = int(sum(slice))

        liquid_indices_in_array = numpy.array(liquid.liquid)

        connect_list = liquid_indices_in_array[boo1].tolist()
        nest.DivergentConnect([inputs[i]],connect_list,weight = weight, delay = delay)
        #for j in range(len(liquid.liquid)):
        #    if numpy.random.rand() <= connectivity:
        #       nest.Connect([inputs[i]],[liquid.liquid[j]],{'weight':weight,'delay':delay})
        #       connect_list.append(liquid.liquid[j])
        input_connect_list.append(connect_list)
    return input_connect_list



def training_slice_evolution(liquid,sim_time=500.0,bin_size=0.1,tau=2.0):
    """
       This function returns the prominence of each training slice in the liquid's spatio-temporal response
       The returned matrix, Evolution, will have n_spike_patterns rows and L columns, each column being a time bin.
       Remember to keep bin_size small enough such that network_state has only 1's and 0's
    """
    spike_state = nest.GetStatus(liquid.spike_detectors, 'events')
    network_state = []

    pattern = liquid.linkpattern
    n_spike_patterns = liquid.n_spike_patterns

    bins = numpy.arange(sim_time/bin_size+1)*bin_size # the elements here "bracket" the bins
    bincenters = bins[0:-1] + ( bin_size / 2.0 )

    for i in liquid.exc_neurons:  # enumerate over every exc. neuron (dimension)

        # I need to know the index of these exc. neurons within the list of network neurons
        nrn = liquid.liquid.index(i)
        spikes = spike_state[nrn].get("times") # get all the spike times from this particular neuron
        t = numpy.zeros(len(bins)-1) # the actual number of bins

        if len(spikes):
           t = numpy.histogram(spikes, bins, new=True)[0]
        network_state.append(t)

    network_state = numpy.array(network_state)

    mn = network_state.shape
    #print network_state
    L = mn[1] # this is the number of times we sample the prominence of a particular pattern in the network
    #print L
    Evolution = numpy.zeros([n_spike_patterns,L]) # this is the solution we seek

    m = numpy.zeros(n_spike_patterns) # initialize my pattern state variable
    n = 0.0 # initialize my global state variable

    for j in range(L): # enumerate over the time bins

        nstate = network_state[:,j]
        q = nstate > 0
        n = n * numpy.exp(-bin_size/tau) + len(nstate[q]) # the network_state variable n at this time bin

        for i in range(n_spike_patterns):

            p = pattern[:,i] > 0
            m[i] = m[i] * numpy.exp(-bin_size/tau) + len( p[q][p[q]] ) # p[q][p[q]] gives the overlap of firing neurons
            M = sum(pattern[:,i])
            #if (i==0):
               #print i,j
               #print m[i],n,M,liquid.ntotal,len(nstate),len(nstate>0)
               #print p
               #print q
               #print p[q][p[q]]

            Evolution[i,j] = m[i] / numpy.float(M) - n / numpy.float(liquid.ntotal)

    #print Evolution

    return Evolution, bincenters



def write_raster(liquid, raster_file, input_spike_times, input_connect_list):
    """
       This is a routine that writes data related to my raster plots to an output file with id "raster_file"
    """
    # here is where I write data to the raster data file
    # the data will be written with one row of spike times, followed by one row of neuron indices
    group1_spikes = nest.GetStatus(liquid.group1_spike_detector,'events')
    senders = group1_spikes[0]['senders']
    times = group1_spikes[0]['times']
    L = len(times)
    LL = numpy.array([L])
    #raster_file.write('\n')
    LL.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')
    spike_times = numpy.array(times)
    spike_times.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')
    spike_senders = numpy.array(senders)
    spike_senders.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')

    #raster_file.write('\n')
    ## now the same for the inhibitory neurons
    inh_network_spikes = nest.GetStatus(liquid.group2_spike_detector,'events')
    inh_senders = inh_network_spikes[0]['senders']
    inh_times = inh_network_spikes[0]['times']
    L = len(inh_times)
    LL = numpy.array([L])
    LL.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')
    inhtimes = numpy.array(inh_times)
    inhtimes.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')
    inhsenders = numpy.array(inh_senders)
    inhsenders.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')

    ## now the inputs
    [input_times,input_senders] = Input_Scatter(input_spike_times,input_connect_list)
    L = len(input_times)
    LL = numpy.array([L])
    LL.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')
    inptimes = numpy.array(input_times)
    inptimes.tofile(raster_file,sep=' ',format="%s")
    raster_file.write('\n')
    inpsenders = numpy.array(input_senders)
    inpsenders.tofile(raster_file,sep=' ',format="%s")




def all_bins_psth_in_classes(liquid, sim_time, time_bin, gap_endpoints, number_of_bins, class_labels):
    """
    """
    indices0 = class_labels == 0
    indices1 = class_labels == 1
    for i in range(number_of_bins):
        psth = Pick_Liquid_Gap_PSTH(liquid, sim_time, time_bin, gap_endpoints, i*time_bin)
        if i == 0:
           all_psth0 = psth[:,indices0]
           all_psth1 = psth[:,indices1]
        else:
           all_psth0 = numpy.concatenate((all_psth0,psth[:,indices0]))
           all_psth1 = numpy.concatenate((all_psth1,psth[:,indices1]))

    return all_psth0, all_psth1
