#!/usr/bin/env python

import numpy as np
from pymtl.misc import numerics
from pymtl.interfaces.mtl_priors import LowRankGaussianParams
from pymtl.mtl_linear_regression import TemporalBRC
import scipy.io as scio
import scipy.optimize as optimize
import sympy
import matplotlib.pyplot as plt
from scipy.signal import freqz

def test_low_rank_approx(fun, n, k):
    ''' 
    Function that takes a function, a n, and a k, and uses that function
    to generate samples from an n-dimensional vector, then compares the empirical mean
    and the computed low-rank mean and prints the rank of the approximation as well 
    as the squared error.

    for now as many samples as dimensions
    '''
    dim = len(numerics.vech(np.eye(n)))
    samples = [np.random.rand(dim,1) for i in range(10)]
    sample_mean = sum(samples)/n
    fun_mean = fun(samples)
    fun_diff = sum([np.abs(s - fun_mean) for s in samples])
    mean_diff = sum([np.abs(s - sample_mean) for s in samples])
    
    print('mean difference/ mean: {}/{}'.format(np.mean(fun_diff),
                                                        np.mean(mean_diff)))
    U = numerics.unvech(fun_mean)
    print('Rank of prior matrix: {}/{}'.format(np.linalg.matrix_rank(U),U.shape[0]))    
    
def test_LRGP(n,k):
    dim = len(numerics.vech(np.eye(n)))
    model = LowRankGaussianParams(dim,k=k,nu=10,estimator='LedoitWolf')
    def testfun(samp):
        model.update_params(samp)
        return model.mu
    test_low_rank_approx(testfun, n, k)

def test_vectranspose():
    test = np.arange(10).reshape(5,2)
    Tnk = numerics.generate_vectranspose_matrix(5,2)
    assert (test.T.reshape((-1,1)) == Tnk.dot(test.reshape((-1,1)))).all(), 'vectranspose not working properly'

def test_elim_dup(f_D, f_E):
    dim = 12
    test = np.arange(dim**2).reshape((dim,dim))
    assert (f_E(test) == f_E(f_D(f_E(test)))).all(), 'Error in vech/unvech'

def test_temporal_learning(flen=[5]):
    '''
    Function that computes CSP filters then uses those with the temporal filter MTL 
    idea, and confirms that the output has a spectral profile that is similar to expected.
    Generate y values from trace of filtered covariance to ensure that is not an issue
    '''
    def covmat(x,y):
        return (1/(x.shape[1]-1)*x.dot(y.T))
    D = scio.loadmat('/is/ei/vjayaram/code/python/pyMTL_MunichMI.mat')
    data = D['T'].ravel()
    labels = D['Y'].ravel()
    fig = plt.figure()
    fig.suptitle('Recovered frequency filters for various filter lengths')
    model = TemporalBRC(max_prior_iter=100)
    # compute cross-covariance matrices and generate X
    for find,freq in enumerate(flen):
        X = []
        for tind,d in enumerate(data):
            d = d.transpose((2,0,1))
            x = np.zeros((d.shape[0], freq))
            nsamples = d.shape[2]
            for ind, t in enumerate(d):
                for j in range(freq):
                    C = covmat(t[:,0:(nsamples-j)],t[:,j:])
                    x[ind,j] = np.trace(C + C.T)
            X.append(x)
            labels[tind] = labels[tind].ravel()

        model.fit_multi_task(X,labels)
        b = numerics.solve_fir_coef(model.prior.mu.flatten())[0]
        # look at filter properties
        w,h = freqz(b[1:],worN=100)
        w = w*500/2/np.pi # convert to Hz

        ax1 = fig.add_subplot(3,3,find+1)
        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        plt.plot(w, angles, 'g')
        plt.ylabel('Angle (radians)', color='g')
        plt.grid()
        plt.axis('tight')
    plt.show()

if __name__ == "__main__":
    #test_elim_dup(numerics.unvech, numerics.vech)
    #test_LRGP(30,5)
    test_temporal_learning([10])
    
