import pandas as pd
import numpy as np

class Indicator:
    """
    Indicator set class for uniqueness computation and sequential bootstrap.
    """
    def __init__(self, bar_series):

        ## initialize set of intervals
        self.iset = list(bar_series.iteritems())

        ## initialize array of concurrent values
        self.conc = np.zeros(max(bar_series) + 1)
        for (t0,t1) in self.iset:
            self.conc[range(t0, t1 + 1)] += 1

    def total_avg_uniqueness(self):
        """
        Returns avg uniqueness for the entire set
        """
        ux = np.array(self.__avg_uniq(self.iset, self.conc))
        return np.mean(ux[ux > 0])

    def sample_avg_uniqueness(self, which):
        """
        Returns average uniqueness for a specific set of values
        """
        s_iset = [ self.iset[i] for i in which ]
        s_conc = np.zeros(self.conc.shape[0])
        for (t0,t1) in s_iset:
            s_conc[range(t0, t1 + 1)] += 1

        ux = np.array(self.__avg_uniq(s_iset, s_conc))
        return np.mean(ux[ux > 0])        

    def standard_bootstrap(self, size=None):
        """
        Returns a conventional bootstrap
        """
        if size is None:
            size = len(self.iset)
        return list(np.random.randint(size, size=size))
    
    def sequential_bootstrap(self, size=None):
        """
        Returns a sequential bootstrap.
        """
        if size is None:
            size = len(self.iset)

        phi = []
        s_iset = []
        s_conc = np.zeros(self.conc.shape[0])
        while len(phi) < size:

            ## pick a random entry
            ix = np.random.randint(0, len(self.iset))

            ## compute its average uniqueness if added to the current set
            s_t0,s_t1 = self.iset[ix]
            ux = np.mean(1.0 / (s_conc[range(s_t0, s_t1 + 1)] + 1))
            vx = np.random.rand()

            ## accept using a biased coin
            if (vx < ux):
                phi.append(ix)
                s_iset.append((s_t0, s_t1))
                s_conc[range(s_t0, s_t1 + 1)] += 1
        
        return phi
    
    def __avg_uniq(self, _iset, _conc):
        return [ np.mean(1.0 / _conc[range(t0,t1 + 1)]) for (t0,t1) in _iset ]
