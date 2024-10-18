# in collaboration with Nikolay Chenkov of BCCN Berlin
class Perceptron:
    """
        class for the perceptron,
        contains all the data and function for the classification task
    """
    def __init__(self, psth, max_iter=5000):
        """
        init the perceptron
        """       
        self.psth = psth
        self.max_iter = max_iter
        self.weights = numpy.random.randn(len(psth[0][0]))        
        self.alpha = .001
        self.max_counter = 1000   # if max_counter correct outputs then learned


    def train(self):
        """ perceptron training function"""
        counter = 0
        for iter in range(self.max_iter):
            # pick a random class (0 or 1)
            t_cls = ((numpy.sign(numpy.random.randn()) +1 )/2).astype('Int32')
            # pick a random stimuli from the class
            t_trn = numpy.random.randint(len(self.psth[t_cls])) 
            psth = self.psth[t_cls][t_trn]
            # pick a random time step of that stimuli
            t_time_s  = numpy.random.randint(psth.shape[1])            
            u = self.update_weight(psth[:,t_time_s], t_cls)
            if u: counter += 1
            else: counter = 0
            # stop if there was no error in the last ... trials
            if counter > self.max_counter: break
                        
                        
    def update_weight(self, psth, t_cls):
        """ updates the weight according to the psth"""
        p_out = self.output(psth)
        if t_cls != p_out:
            d_w = self.alpha*(t_cls - p_out)*psth
            self.weights +=d_w
            return 0
        return 1
            
            
    def output(self, psth):
        """ calculates a perceptron output"""
        sign = numpy.sign(sum(psth*self.weights))
        if not sign: return 1.
        return (sign+1.)/2.
    
    
    def check_correctness(self):
        """ check for the corectness of preceptrons
        going through the whole data"""     
        corr = 0
        iter = 0        
        for t_cls in range(2):                   
            for t_trn in range(len(self.psth[t_cls])):
                psth = self.psth[t_cls][t_trn]                
                for i in range(psth.shape[1]):
                    iter+=1
                    p_out= self.output(psth[:,i])
                    if p_out==t_cls:
                        corr += 1.
        perc_corr = corr/iter
        print 'percentage correct classificication: ', perc_corr*100.
        return perc_corr


if __name__=='__main__':

   print 'boo!'
