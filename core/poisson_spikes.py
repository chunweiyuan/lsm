### a script for poisson spike generation, courtesy of Nikolay Chenkov from BCCN Berlin
import numpy
from matplotlib import pyplot
import random
import pickle
#from scipy import signal

def gen_spike_train(sim_time=1000., rate = 100., time_res = .1):
    """
        create a poissonian spike train 
        time and time resolution in ms
        rate in Hz
    """
    u = poisson_spikes(sim_time, rate)/time_res  # generate poisson spikes
    while not len(u):                       # which is not empty
        u = poisson_spikes(sim_time, rate)/time_res   
    spikes_u = numpy.zeros(sim_time/time_res)
    spikes_u[u.astype('Int32')] = 1.    # from spike time to spike train   
    #spikes_u = signal.convolve(spikes_u, g, 'same') # convolve with gaussian        
    return spikes_u
      

def multiple_trains(number_trains=1, time=500., rate=20.):
    """
        returns list with number_trains spike times
    """
    train_list = []
    for i in range(number_trains):
        train_list.append(poisson_spikes(time, rate))
    
    return train_list

def poisson_spikes(duration=1000., rate = 100.):
    """
        returns array with spike times
        time in ms
        rate in Hz
    """
    
    # produce 3 times more spikes than the expected value
    # this is just to guard against statistical fluctuation, so that I have enough spikes for the later cut-off in command:
    # --> spike_times = numpy.cumsum(ISIs)
    n = 3.*rate*duration/1000.
    
    # so this is the number of spikes I generate
    randNumbers = numpy.random.rand(n)

    ISIs = -numpy.log(randNumbers)/rate
    ISIs = ISIs*1000.
    
    spike_times = numpy.cumsum(ISIs)
    return spike_times[spike_times<duration]

def test_1():
    
    time=10000000.
    rate = 100.

    spk = poisson_spikes(time, rate)
    
    print len(spk)*1000./time
    print spk
    pyplot.hist(spk[0:100],1000)
    pyplot.show()
    
def mes(a,b,t_s):
    """ calculate euclidian distance between 2 spike trains (vectors)"""
    d = ((sum((a-b)**2.))**.5)/50.
    return d
    
    
def create_spiketrain_couples():        
    sim_time = 500.
    t_s = .1
    rate = 10.
    
    # empty lists for the spike train storage
    list_spikes_0 = []
    list_spikes_1 = []
    list_spikes_2 = []
    list_spikes_4 = []
    
    #define a gaussian for the convolution 
    tao = 50
    g_width = 8*tao
    t = numpy.arange(g_width) - g_width/2.
    g = numpy.exp(-(t/tao)**2.)

    
    for i in range(5000):
        rate = 12.*numpy.random.rand() + 5.  # define some f. rate
        u = poisson_spikes(sim_time, rate)/t_s  # generate poisson spikes
        while not len(u):
            u = poisson_spikes(sim_time, rate)/t_s
        spikes_u = numpy.zeros(sim_time/t_s)
        spikes_u[u.astype('Int32')] = 1.    # from spike time to spike train
        spikes_u = signal.convolve(spikes_u, g, 'same') # convolve with gaussian
        
        v = u + random.gauss(0.,30.)
        v = v[v<sim_time]
        spikes_v = numpy.zeros(sim_time)
        spikes_v[v.astype('Int32')] = 1.    # from spike time to spike train
        spikes_v = signal.convolve(spikes_v, g, 'same') # convolve with gaussian
        
        dist = mes(spikes_u,spikes_v,t_s) # calc distace
        #print 'diff %.3f '% dist

        if (dist <.01):  
            list_spikes_0.append([u,v])
            
        if len(list_spikes_0)>=200:
            break
    
    for i in range(50000):
        rate = 15.*numpy.random.rand() + 10.  # define some f. rate 
        u = poisson_spikes(sim_time, rate)  # generate poisson spikes
        spikes_u = numpy.zeros(sim_time)
        spikes_u[u.astype('Int32')] = 1.    # from spike time to spike train
        spikes_u = signal.convolve(spikes_u, g, 'same') # convolve with gaussian
        
        r = random.gauss(rate,rate/2.)       # define a slidely different rate 
        v = poisson_spikes(sim_time, abs(r)) # generate poisson spikes
        spikes_v = numpy.zeros(sim_time)
        spikes_v[v.astype('Int32')] = 1.    # from spike time to spike train
        spikes_v = signal.convolve(spikes_v, g, 'same') # convolve with gaussian
        
        dist = mes(spikes_u,spikes_v,t_s) # calc distace
        print 'diff %.3f '% dist

        if (dist <.11 and dist >.09 and len(list_spikes_1)<200):
            list_spikes_1.append([u,v])
            continue
            
        if (dist <.21 and dist>.19 and len(list_spikes_2)<200):
            list_spikes_2.append([u,v])
            continue

 
    for i in range(50000):
        rate = 25.*numpy.random.rand() + 20.  # define some f. rate 
        u = poisson_spikes(sim_time, rate)  # generate poisson spikes
        spikes_u = numpy.zeros(sim_time)
        spikes_u[u.astype('Int32')] = 1.    # from spike time to spike train
        spikes_u = signal.convolve(spikes_u, g, 'same') # convolve with gaussian
        
        r = random.gauss(rate,rate/4.)       # define a slidely different rate 
        v = poisson_spikes(sim_time, abs(r)) # generate poisson spikes
        spikes_v = numpy.zeros(sim_time)
        spikes_v[v.astype('Int32')] = 1.    # from spike time to spike train
        spikes_v = signal.convolve(spikes_v, g, 'same') # convolve with gaussian
        
        dist = mes(spikes_u,spikes_v,t_s) # calc distace
        print 'diff %.3f '% dist

        if (dist <.41 and dist>.39):
            list_spikes_4.append([u,v])           

        if len(list_spikes_4)>=200:
            break
        
    #print u
    #pyplot.subplot(211)
    #pyplot.plot(spikes_u)
    #pyplot.subplot(212)
    #pyplot.plot(spikes_v)
    #pyplot.show()

    
    
    print 'size of the spike train couples: '
    print len(list_spikes_0), len(list_spikes_1), len(list_spikes_2), len(list_spikes_4)

    filehandler = open('list_spikes_0', 'w')
    pickle.dump(list_spikes_0, filehandler) 

    filehandler = open('list_spikes_1', 'w')
    pickle.dump(list_spikes_1, filehandler) 
    
    filehandler = open('list_spikes_2', 'w')
    pickle.dump(list_spikes_2, filehandler) 
    
    filehandler = open('list_spikes_4', 'w')
    pickle.dump(list_spikes_4, filehandler) 
    
if __name__=='__main__':
    
    #test_1()   # test function

    #create_spiketrain_couples()

    s = gen_spike_train(sim_time=1000., rate = 100., time_res = .1)
    print sum(s), s.mean(), len(s)
    
