# Copyright 2014-2019 Anton Sobinov

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from itertools import izip
import numpy


def rms(u, v):
    """First one - data, second - calculated"""
    if not isinstance(u, numpy.ndarray):
        u = numpy.array(u)
    if not isinstance(v, numpy.ndarray):
        v = numpy.array(v)
    return math.sqrt(numpy.mean((u-v)**2))
    # return math.sqrt(sum((ui-vi)**2 for ui, vi in izip(u, v)) / len(u))


def rms_n2am(u, v):
    """First one - data, second - calculated. Normalized to abs max u"""
    if not isinstance(u, numpy.ndarray):
        u = numpy.array(u)
    if not isinstance(v, numpy.ndarray):
        v = numpy.array(v)
    top = rms(u, v)
    bottom = abs(u).max()
    if top == bottom == 0:
        return 0.
    return top / bottom


def rms_n2mv(u, v):
    """First one - data, second - calculated. Normalized to middle u

    Assume u positive.
    """
    if not isinstance(u, numpy.ndarray):
        u = numpy.array(u)
    if not isinstance(v, numpy.ndarray):
        v = numpy.array(v)
    top = rms(u, v)
    bottom = u.max() + u.min()
    if top == bottom == 0:
        return 0.
    return 2*top / bottom


def rms_n2rom(u, v):
    """First one - data, second - calculated. Normalized to range of u

    Assume u positive.
    """
    if not isinstance(u, numpy.ndarray):
        u = numpy.array(u)
    if not isinstance(v, numpy.ndarray):
        v = numpy.array(v)
    top = rms(u, v)
    bottom = u.max() - u.min()
    if top == bottom == 0:
        return 0.
    return top / bottom


def nrms(u, v):
    """First one - data, second - calculated

    u does not go through 0
    """
    if sum(abs(ui)+abs(vi) for ui, vi in izip(u, v)) == 0:
        return 0.
    return math.sqrt(sum((1.-vi/ui)**2 for ui, vi in izip(u, v)) / len(u))


def ITC(dim, metric, ndata, sigma=math.e):
    """Returns value of Information Tradeoff Criterion

    Evaluates
        dim + math.log(metric, sigma)
    The lower ITC - the better. Consider two approximations that produce
    metrics N1 and N2, while N1 has k less parameters. Then N2 will be better
    if and only if
        (sigma ** k) * N2 < N1

    Parameters:
        dim - amount of parameters of the approximation
        metric - value of a metric on the approximation
        sigma - tradeoff criterion, defaults to exponent
    """
    if metric == 0:
        return -numpy.inf
    return dim + math.log(metric, sigma)


def AIC(dim, metric, ndata, sigma=math.e):
    """Returns value of Akaike Information Criterion

    Evaluates
        2*dim + 2*ndata*math.log(metric, sigma)
    The lower AIC - the better the model. For more information try
    https://en.wikipedia.org/wiki/Akaike_information_criterion

    Parameters:
        dim - amount of parameters of the approximation
        metric - value of a metric on the approximation (RMS)
        ndata - number of data points that was used for metric estimation
        sigma - tradeoff criterion, defaults to exponent
    """
    if metric == 0:
        return -numpy.inf
    return 2.*dim + 2.*ndata*math.log(metric, sigma)


def AICc(dim, metric, ndata, sigma=math.e):
    """Returns value of corrected Akaike Information Criterion

    Evaluates
        2*dim + 2*ndata*math.log(metric, sigma) + 2*dim*(dim+1)/(ndata-dim-1)
    The lower AIC - the better the model. For more information try
    https://en.wikipedia.org/wiki/Akaike_information_criterion

    Parameters:
        dim - amount of parameters of the approximation
        metric - value of a metric on the approximation (RMS)
        ndata - number of data points that was used for metric estimation
        sigma - tradeoff criterion, defaults to exponent
    """
    if metric == 0:
        return -numpy.inf
    if dim >= ndata - 1:
        return numpy.inf
    correction = 2.*dim*(dim+1.)/(ndata-dim-1.)
    return AIC(dim, metric, ndata, sigma=sigma) + correction
