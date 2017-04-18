#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: GMM.PY
Date: Friday, June 24 2011/Volumes/NO NAME/seds/nodes/gmm.py
Description: A python class for creating and manipulating GMMs.
"""

import scipy.cluster.vq as vq
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
import tqdm

npa = np.array

from imp import reload

import sys;

sys.path.append('.')

import pylab
from normal import Normal


class GMM(object):
    def __init__(self, dim=None, ncomps=None, data=None, method=None, filename=None, params=None):

        self.nanfill = False

        if not filename is None:  # load from file
            self.load_model(filename)

        elif params is not None:  # initialize with parameters directly
            self.comps = params['comps']
            self.ncomps = params['ncomps']
            self.dim = params['dim']
            self.priors = params['priors']

        elif data is not None:  # initialize from data

            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps
            self.comps = []

            if method is "uniform":
                # uniformly assign data points to components then estimate the parameters
                npr.shuffle(data)
                n = len(data)
                s = n / ncomps
                for i in range(ncomps):
                    self.comps.append(Normal(dim, data=data[i * s: (i + 1) * s]))

                self.priors = np.ones(ncomps, dtype="double") / ncomps

            elif method is "random":
                # choose ncomp points from data randomly then estimate the parameters
                mus = pr.sample(data, ncomps)
                clusters = [[] for _ in range(ncomps)]
                for d in data:
                    i = np.argmin([la.norm(d - m) for m in mus])
                    clusters[i].append(d)

                for i in range(ncomps):
                    self.comps.append(Normal(dim, mu=mus[i], sigma=np.cov(clusters[i], rowvar=0)))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            elif method is "nanfill":
                # choose standard normal as init point
                mus = []
                while True:
                    mu = pr.sample(data, 1)[0]
                    if not np.any(np.isnan(mu)):
                        mus.append(mu)
                        if len(mus) == ncomps:
                            break
                clusters = [[] for _ in range(ncomps)]
                for d in data:
                    i = np.argmin([la.norm(d - m) for m in mus])
                    clusters[i].append(d)

                mus = npa(mus)

                for i in range(ncomps):
                    self.comps.append(Normal(dim, mu=mus[i],
                                             sigma=np.diag(np.ones(dim, dtype='float64'))))

                self.priors = np.ones(ncomps, dtype="double") / ncomps
                self.nanfill = True

            elif method is "nanfill2":
                # slightly smarter way for init
                means = []
                stds = []
                X = data
                na = np.logical_or(np.isnan(data), np.isinf(data)).T
                for k in range(data.shape[1]):
                    temp = X[~na[k], k]
                    assert not np.any(np.isnan(temp)) and not np.any(np.isinf(temp))
                    means.append(np.mean(temp))
                    stds.append(np.std(temp))
                means = np.array(means)
                stds = np.array(stds)

                filt = np.logical_or(np.isnan(stds), np.isinf(stds), stds<0)
                stds[filt] = np.nanmean(stds)

                for i in range(ncomps):
                    self.comps.append(Normal(dim, mu=means + np.random.normal(scale=np.nanmean(stds)/3, size=means.shape),
                                             sigma=np.diag(np.array([np.mean(stds)]*len(stds)))))

                self.priors = np.ones(ncomps, dtype="double") / ncomps
                self.nanfill = True


            elif method is "kmeans":
                # use kmeans to initialize the parameters
                (centroids, labels) = vq.kmeans2(data, ncomps, minit="points", iter=100)
                clusters = [[] for i in range(ncomps)]
                for (l, d) in zip(labels, data):
                    clusters[l].append(d)

                # will end up recomputing the cluster centers
                for cluster in clusters:
                    self.comps.append(Normal(dim, data=cluster))

                self.priors = np.ones(ncomps, dtype="double") / np.array([len(c) for c in clusters])

            else:
                raise ValueError, "Unknown method type!"

        else:

            # these need to be defined
            assert dim and ncomps, "Need to define dim and ncomps."

            self.dim = dim
            self.ncomps = ncomps

            self.comps = []

            for i in range(ncomps):
                self.comps.append(Normal(dim))

            self.priors = np.ones(ncomps, dtype='double') / ncomps

    def __str__(self):
        res = "%d" % self.dim
        res += "\n%s" % str(self.priors)
        for comp in self.comps:
            res += "\n%s" % str(comp)
        return res

    def save_model(self):
        pass

    def load_model(self):
        pass

    def mean(self):
        return np.sum([self.priors[i] * self.comps[i].mean() for i in range(self.ncomps)], axis=0)

    def covariance(self):  # computed using Dan's method
        m = self.mean()
        s = -np.outer(m, m)

        for i in range(self.ncomps):
            cm = self.comps[i].mean()
            cvar = self.comps[i].covariance()
            s += self.priors[i] * (np.outer(cm, cm) + cvar)

        return s

    def pdf(self, x):
        responses = [comp.pdf(x) for comp in self.comps]
        return np.dot(self.priors, responses)

    def condition(self, indices, x):
        """
        Create a new GMM conditioned on data x at indices.
        """
        condition_comps = []
        marginal_comps = []

        for comp in self.comps:
            condition_comps.append(comp.condition(indices, x))
            marginal_comps.append(comp.marginalize(indices))

        new_priors = []
        for (i, prior) in enumerate(self.priors):
            new_priors.append(prior * marginal_comps[i].pdf(x))
        new_priors = npa(new_priors) / np.sum(new_priors)

        params = {'ncomps': self.ncomps, 'comps': condition_comps,
                  'priors': new_priors, 'dim': marginal_comps[0].dim}

        return GMM(params=params)

    def restore_nan(self, data):
        restored = []
        for d in data:
            filt = np.logical_or(np.isnan(d), np.isinf(d))
            if np.any(filt):
                indices = np.arange(len(d))[~filt]
                values = d[~filt]
                part = self.condition(indices, values).mean()
                new = []
                count = 0
                for i, k in enumerate(~filt):
                    if k:
                        new.append(d[i])
                    else:
                        new.append(part[count])
                        count += 1
                restored.append(new)
        return np.array(restored)

    def em(self, data, nsteps=100, reg=1e-4):
        '''
        fit model using EM
        :param nsteps: number of steps
        :param reg: regularizer, helps to prevent singularities
        :return: data with filled nans, if any
        '''

        k = self.ncomps
        d = self.dim
        n = len(data)

        mask = np.logical_or(np.isnan(data), np.isinf(data))
        data = np.nan_to_num(data)
        ran = np.arange(mask.shape[1])
        #print (self.priors)

        if not self.nanfill:
            assert not np.any(mask), 'nan filling is supported only with "nanfill" init method'

        for l in tqdm.tqdm(range(nsteps)):

            # E step

            responses = np.zeros((k, n))

            for j in range(n):
                for i in range(k):
                    if self.nanfill and np.any(mask[j]):
                        conditioned = self.comps[i].condition(ran[~mask[j]], data[j][~mask[j]]).mu
                        iter = 0
                        arr = []
                        for h, incl in enumerate(mask[j]):
                            if incl:
                                arr.append(conditioned[iter])
                                iter += 1
                            else:
                                arr.append(0.)
                        arr = npa(arr)
                        unmasked = data[j] * ~mask[j] + arr
                        data[j] = unmasked
                        #if np.any(np.isnan(unmasked)):
                            #print(unmasked)
                        #print(unmasked)
                    else:
                        unmasked = data[j]

                    responses[i, j] = self.priors[i] * self.comps[i].pdf(unmasked)
            #print 'sum resp:'
            #print(np.sum(responses, axis=0))
            responses /= np.sum(responses, axis=0)  # normalize the weights

            # M step

            N = np.sum(responses, axis=1)
            #print(responses)

            for i in range(k):
                mu = np.dot(responses[i, :], data) / N[i]
                sigma = np.zeros((d, d))

                for j in range(n):
                    sigma += responses[i, j] * np.outer(data[j, :] - mu, data[j, :] - mu)

                sigma = sigma / N[i]

                self.comps[i].update(mu, sigma + np.diag(np.ones(len(sigma), dtype='float64'))*reg)  # update the normal with new parameters
                self.priors[i] = N[i] / np.sum(N)  # normalize the new priors
        return self.restore_nan(data)


def shownormal(data, gmm):
    xnorm = data[:, 0]
    ynorm = data[:, 1]

    # Plot the normalized faithful data points.
    fig = pylab.figure(num=1, figsize=(4, 4))
    axes = fig.add_subplot(111)
    axes.plot(xnorm, ynorm, '+')

    # Plot the ellipses representing the principle components of the normals.
    for comp in gmm.comps:
        comp.patch(axes)

    pylab.draw()
    pylab.show()


if __name__ == '__main__':
    """
    Tests for gmm module.
    """

    # x = npr.randn(20, 2)

    # print "No data"
    # gmm = GMM(2,1,2) # possibly also broken
    # print gmm

    # print "Uniform"
    # gmm = GMM(2,1,2,data = x, method = "uniform")
    # print gmm

    # print "Random"
    # gmm = GMM(2,1,2,data = x, method = "random") # broken
    # print gmm

    # print "Kmeans"
    # gmm = GMM(2,1,2,data = x, method = "kmeans") # possibly broken
    # print gmm


    x = np.arange(-10, 30)
    # y = x ** 2 + npr.randn(20)
    y = x + npr.randn(40)  # simple linear function
    # y = np.sin(x) + npr.randn(20)
    data = np.vstack([x, y]).T
    print data.shape

    gmm = GMM(dim=2, ncomps=4, data=data, method="random")
    print gmm
    shownormal(data, gmm)

    gmm.em(data, nsteps=1000)
    shownormal(data, gmm)
    print gmm
    ngmm = gmm.condition([0], [-3])
    print ngmm.mean()
    print ngmm.covariance()
