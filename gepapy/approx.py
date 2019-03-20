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

from functools import partial
import numpy
import scipy.linalg
from copy import deepcopy
from multiprocessing import Pool, cpu_count

import ersatz
import metric


def try_configuration_p(*args, **kwargs):
    """Parallel wrap of try_configuration"""
    if len(args) == 1:  # mapping?
        args = args[0]

    conf, x, y, metric_f, itcf = args

    return try_configuration(conf, x, y, metric_f, itcf=itcf)


def try_configuration(conf, x, y, metric_f, itcf=None):
    """Uses one specific configuration to fit the data

    Parameters:
        see approximate_ascending

    Returns a dictionary:
        see approximate_ascending
    """
    if isinstance(conf, str):
        conf = ersatz.str2dic(conf)
    # if len(conf.keys()) < 1:
    #     return {'config': {}, 'point': [], 'metric': numpy.inf,
    #             'ITC': numpy.inf, 'residuals': numpy.inf,
    #             'conditioned': True}
    if itcf is None:
        itcf = metric.AICc

    dim = len(x[0])
    A = ersatz.make_A(x, conf)
    res = scipy.linalg.lstsq(A, y)

    # Calculate metric and ITC
    ers = ersatz.Ersatz(dim, conf, point=res[0])
    m = metric_f(y, ers(x))
    itc = itcf(len(res[0])+1, m, len(y))

    return {'config': conf, 'point': res[0], 'metric': m,
            'ITC': itc, 'residuals': res[1],
            'conditioned': len(res[0]) == res[2]}


def improve_once(x, y, c_winner, full_dic, metric_f, itcf, min_metr, min_metr_sdif,
                 return_candidates=False, look_ahead=0):
    improving = True
    # construct dics for all possible candidates
    candidates = ersatz.elongate(c_winner['config'], full_dic)
    if len(candidates) < 1:  # nowhere to elongate
        if return_candidates:
            return (False, c_winner, tuple())
        else:
            return (False, c_winner)

    # Calculate how good they fit
    results = []
    for candidate in candidates:
        results.append(try_configuration(candidate, x, y, metric_f, itcf))

    if return_candidates:
        dump = (deepcopy(candidates), deepcopy(results))

    # Sort & move badly conditioned to end:
    factor = max(results, key=lambda x: x['ITC'])['ITC']*1000
    best = min(results,
               key=lambda x: x['ITC'] + factor*int(not x['conditioned']))

    # Assess the best one
    if best['ITC'] < c_winner['ITC']:
        c_winner = best
    else:
        improving = False
        for _ in xrange(look_ahead):
            # check one step ahead
            candidates2 = []
            for candidate in candidates:
                for can in ersatz.elongate(candidate, full_dic):
                    if can not in candidates2:
                        candidates2.append(can)
            candidates = candidates2

            if len(candidates) > 0:
                # Calculate how good they fit
                results = []
                for candidate in candidates:
                    results.append(try_configuration(candidate, x, y, metric_f, itcf))

                if return_candidates:
                    dump = (dump[0]+deepcopy(candidates), dump[1]+deepcopy(results))

                # Sort & move badly conditioned to end:
                factor = max(results, key=lambda x: x['ITC'])['ITC']*1000
                best2 = min(results,
                            key=lambda x: x['ITC'] + factor*int(not x['conditioned']))
                if best2['ITC'] < best['ITC']:
                    best = best2

                # Assess the best one
                if best['ITC'] < c_winner['ITC']:
                    c_winner = best
                    improving = True

    if not improving and min_metr is not None and (
            c_winner['metric'] - best['metric'] > min_metr_sdif and
            c_winner['metric'] > min_metr):
        c_winner = best
        if c_winner['metric'] < min_metr:
            improving = False
        else:
            improving = True

    if return_candidates:
        return (improving, c_winner, dump)
    else:
        return (improving, c_winner)


def _io_p1(x, y, c_winner, full_dic, metric_f, itcf):
    """Generates parameter pool for the parallel unified test of candidates"""
    candidates = ersatz.elongate(c_winner['config'], full_dic)

    params = []
    for candidate in candidates:
        params.append([candidate, x, y, metric_f, itcf])

    return params


def _io_p2(params, results, c_winner, min_metr, min_metr_sdif):
    improving = True
    if len(params) < 1:  # nowhere to elongate
        return (False, c_winner, c_winner)

    # Sort & move badly conditioned to end:
    factor = max(results, key=lambda x: x['ITC'])['ITC']*1000
    best = min(results,
               key=lambda x: x['ITC'] + factor*int(not x['conditioned']))

    # Assess the best one
    if best['ITC'] < c_winner['ITC']:
        c_winner = best
    else:
        improving = False

    return (improving, c_winner, best)



def approximate_ascending_p(*args, **kwargs):
    """Parallel wrap of approximate_ascending"""
    if len(args) == 1:  # mapping?
        args = args[0]

    x, y, metric_f = args[:3]
    if len(args) > 3:
        kwargs['verbose_prefix'] = args[3]
    if len(args) > 4:
        kwargs['rho'] = args[4]

    return approximate_ascending(x, y, metric_f, **kwargs)


def approximate_ascending(x, y, metric_f, itcf=None,
                          min_metr=1e-4, rho=4,
                          min_metr_sdif=None, verbose=0, verbose_prefix='',
                          return_progress=0, look_ahead=0):
    """Approximates y on multidimensional points x using ascending algorithm

    Creates a good enough approximation of data from R^N->R with a set of
    polynomial functions.

    Parameters:
        x: Nxl list, where N is dimensionality of the function.
            len(x) = l, len(x[0]) = N
        y: list, len(y)=l of datapoints
        metric_f: callable, metric_f(y, y1) should return an evaluation of how
            good y1 fits y. For example functions look at metric_f.py. >=0
        itcf: callable ITC evaluator (override one from metric.py).
            itcf(<dim of solution>, <metric value>, <number of datapoints>)
            Defaults to metric.AICc
        min_metr: minimum metric to stop optimization
        rho: maximum power of polynomial. Refer to ersatz.py
        min_metr_sdif: minimum improvement needed if ITC stopped improving, but
            the desired min_metr was not achieved yet. Defaults to min_metr
        verbose: levels of output during search.
        verbose_prefix: string will be added to the beginning of every verbose
            message from the function. Is second to ease up mapping.
        look_ahead: whether to check one more term when the optimization
            finished.

    Returns a dictionary:
        config: dictionary of ersatz parameters. Can be used to create an
            Ersatz class
        point: values for the parameters of the ersatz configuration
        metric: metric
        ITC: ITC
        residuals: residuals
        conditioned: conditioned?
    """
    if min_metr_sdif is None:
        min_metr_sdif = min_metr
    x = numpy.array(x)
    dim = len(x[0])
    full_dic = ersatz.full_dic_from_dim(dim, rho)

    c_winner = try_configuration({}, x, y, metric_f, itcf)

    if return_progress:
        dump = []

    if verbose:
        print '{}Starting fit.'.format(verbose_prefix)
    improving = True
    while improving:
        if return_progress:  # parralelize with pool
            improving, c_winner, dump_buf = improve_once(
                x, y, c_winner, full_dic, metric_f, itcf,
                min_metr, min_metr_sdif, return_candidates=True,
                look_ahead=look_ahead)
            dump.append(dump_buf)
        else:
            improving, c_winner = improve_once(
                x, y, c_winner, full_dic, metric_f, itcf,
                min_metr, min_metr_sdif, look_ahead=look_ahead)

        if verbose and improving:
            print '{}Using {} to fit. Metric: {}, ITC: {}'.format(
                verbose_prefix, ersatz.dic2str(c_winner['config']),
                c_winner['metric'], c_winner['ITC'])

    if verbose:
        print '{}Finished fitting with ersatz {}. Metric: {}, ITC: {}'.format(
            verbose_prefix, ersatz.dic2str(c_winner['config']),
            c_winner['metric'], c_winner['ITC'])

    if isinstance(c_winner['residuals'], (list, tuple, numpy.ndarray)):
        c_winner['residuals'] = numpy.inf

    if return_progress:
        return (c_winner, dump)

    return c_winner


def approximate_diffgroup_ascending(
        x, y, yd, metric_f, metricd_f, itcf=None,
        min_metr=1e-4, rho=4,
        min_metr_sdif=None, verbose=0, verbose_prefix='',
        return_progress=0, look_ahead=0):
    """Approximates data representing a system of PDE

    System of partial differential equations within the data:
        f(x) = y
        df(x)/dx_i = yd[i]
    is approximated with polynomials. Functions for f(x) and df(x)/dx_i are
    returned.

    Parameters:
        x: Nxl list, where N is dimensionality of the function.
            len(x)=l, len(x[0])=N
        y: list, len(y)=l, integral function
        yd: lxN list, len(y)=N, differential function
        metric_f: callable, metric_f(y, y1) should return an evaluation of how
            good y1 fits y. For example functions look at metric_f.py. >=0
        metricd_f: callable, metrics_f(y, y1) should return an evaluation of
            how good yd1 fits yd. For example functions look at metric_f.py.
            >=0 Specified for the partial differential datapoints.
        itcf: callable ITC evaluator (override one from metric.py).
            itcf(<dim of solution>, <metric value>, <number of datapoints>)
            Defaults to metric.AICc
        min_metr: minimum metric value to stop optimization
        rho: maximum power of integral polynomial. Refer to ersatz.py
        min_metr_sdif: minimum improvement needed if ITC stopped improving, but
            the desired min_metr was not achieved yet. Defaults to min_metr
        verbose: levels of output during search.
        verbose_prefix: string will be added to the beginning of every verbose
            message from the function. Is second to ease up mapping.
        look_ahead: whether to check one more term when the optimization
            finished.

    Returns a list of dictionaries:
        (y, (dy_i, ))
    where each dictionary contains:
        config: dictionary of ersatz parameters. Can be used to create an
            Ersatz class
        point: values for the parameters of the ersatz configuration
        metric: metric
        ITC: ITC
        residuals: residuals
        conditioned: conditioned?
    """
    if min_metr_sdif is None:
        min_metr_sdif = min_metr
    x = numpy.array(x)
    dim = len(x[0])
    full_dic = ersatz.full_dic_from_dim(dim, rho)
    full_dic_d = ersatz.full_dic_from_dim(dim, rho-1)

    c_winner = try_configuration({}, x, y, metric_f, itcf)
    c_winner_d = [try_configuration({}, x, y, metricd_f, itcf)]*dim

    if return_progress:
        dump = []
        dump_m = [[] for _ in xrange(dim)]

    if verbose:
        print '{}Starting fit.'.format(verbose_prefix)

    improving = True
    improving_d = [True]*dim

    while improving or True in improving_d:
        # Calculate nex improvement steps for function and diffs
        if return_progress and improving:
            improving, c_winner, dump_buf = improve_once(
                x, y, c_winner, full_dic, metric_f, itcf,
                min_metr, min_metr_sdif, return_candidates=True,
                look_ahead=look_ahead)
            dump.append(dump_buf)
        elif improving:
            improving, c_winner = improve_once(
                x, y, c_winner, full_dic, metric_f, itcf,
                min_metr, min_metr_sdif, look_ahead=look_ahead)


        if return_progress:
            for i in xrange(dim):
                if improving_d[i]:
                    improving_d[i], c_winner_d[i], dump_buf = improve_once(
                        x, yd[i], c_winner_d[i], full_dic_d, metricd_f, itcf,
                        min_metr, min_metr_sdif,
                        return_candidates=True, look_ahead=look_ahead)
                    dump_m[i].append(dump_buf)
        else:
            for i in [j for j in xrange(dim) if improving_d[j]]:
                improving_d[i], c_winner_d[i] = improve_once(
                    x, yd[i], c_winner_d[i], full_dic_d, metricd_f, itcf,
                    min_metr, min_metr_sdif, look_ahead=look_ahead)

        # Integrate and differentiate
        cwds = [ersatz.Ersatz(dim, cwd['config'])
                for cwd in c_winner_d]
        cw = ersatz.Ersatz(dim, c_winner['config'])

        for i, cwd in enumerate(cwds):
            cw |= cwd.integral(i)
        cwds = cw.differential()

        # calculate the vals
        c_winner_d = [
            try_configuration(cwd.config_dic, x, yd[i], metricd_f)
            for i, cwd in enumerate(cwds)]
        c_winner = try_configuration(cw.config_dic, x, y, metric_f)

        if verbose and improving:
            print '{}Using {} to fit. Metric: {}, ITC: {}'.format(
                verbose_prefix, ersatz.dic2str(c_winner['config']),
                c_winner['metric'], c_winner['ITC'])

    if verbose:
        print '{}Finished fitting with ersatz {}. Metric: {}, ITC: {}'.format(
            verbose_prefix, ersatz.dic2str(c_winner['config']),
            c_winner['metric'], c_winner['ITC'])

    if isinstance(c_winner['residuals'], (list, tuple, numpy.ndarray)):
        c_winner['residuals'] = numpy.inf
    for i in xrange(dim):
        if isinstance(c_winner_d[i]['residuals'], (list, tuple, numpy.ndarray)):
            c_winner_d[i]['residuals'] = numpy.inf

    if return_progress:
        return (c_winner, c_winner_d, dump, dump_m)

    return (c_winner, c_winner_d)


def approximate_diffgroup_ascending_p(
        x, y, yd, metric_f, metricd_f, pps=None, itcf=None,
        min_metr=1e-4, rho=4,
        min_metr_sdif=None, verbose=0, verbose_prefix='',
        return_progress=0, look_ahead=0):
    """Approximates data representing a system of PDE

    Parallelizes some of the calculations
    System of partial differential equations within the data:
        f(x) = y
        df(x)/dx_i = yd[i]
    is approximated with polynomials. Functions for f(x) and df(x)/dx_i are
    returned.

    Parameters:
        x: Nxl list, where N is dimensionality of the function.
            len(x)=l, len(x[0])=N
        y: list, len(y)=l, integral function
        yd: lxN list, len(y)=N, differential function
        metric_f: callable, metric_f(y, y1) should return an evaluation of how
            good y1 fits y. For example functions look at metric_f.py. >=0
        metricd_f: callable, metrics_f(y, y1) should return an evaluation of
            how good yd1 fits yd. For example functions look at metric_f.py.
            >=0 Specified for the partial differential datapoints.
        pps: int, number of processes to spawn. By default estimates from
            dimensionality of data.
        itcf: callable ITC evaluator (override one from metric.py).
            itcf(<dim of solution>, <metric value>, <number of datapoints>)
            Defaults to metric.AICc
        min_metr: minimum metric value to stop optimization
        rho: maximum power of integral polynomial. Refer to ersatz.py
        min_metr_sdif: minimum improvement needed if ITC stopped improving, but
            the desired min_metr was not achieved yet. Defaults to min_metr
        verbose: levels of output during search.
        verbose_prefix: string will be added to the beginning of every verbose
            message from the function. Is second to ease up mapping.
        look_ahead: whether to check one more term when the optimization
            finished.

    Returns a list of dictionaries:
        (y, (dy_i, ))
    where each dictionary contains:
        config: dictionary of ersatz parameters. Can be used to create an
            Ersatz class
        point: values for the parameters of the ersatz configuration
        metric: metric
        ITC: ITC
        residuals: residuals
        conditioned: conditioned?
    """
    if min_metr_sdif is None:
        min_metr_sdif = min_metr
    x = numpy.array(x)
    dim = len(x[0])
    full_dic = ersatz.full_dic_from_dim(dim, rho)
    full_dic_d = ersatz.full_dic_from_dim(dim, rho-1)

    c_winner = try_configuration({}, x, y, metric_f, itcf)
    cl_winner = try_configuration({}, x, y, metric_f, itcf)
    c_winner_d = [try_configuration({}, x, y, metricd_f, itcf)]*dim
    cl_winner_d = [0]*dim

    if return_progress:
        dump = []
        dump_m = [[] for _ in xrange(dim)]
        if return_progress > 2:
            dump_l = []

    if pps is None:
        pps = min(cpu_count(), dim*dim);

    if verbose:
        print '{}Starting fit with {} process pool.'.format(
            verbose_prefix, pps)
    p = Pool(pps);

    improving = True
    improving_d = [True]*dim

    while improving or True in improving_d:
        if return_progress == 1 and len(dump) == 0:
            dump.append(c_winner)
            for i in xrange(dim):
                dump_m[i].append(c_winner_d[i])

        js_improving = False  # just stopped improving
        js_improving_d = [False]*dim
        # Calculate parameters for one-term pool
        params = []
        params_sl = []
        if improving:  # muscle length
            candidates = ersatz.elongate(c_winner['config'], full_dic)

            params_buf = []
            for candidate in candidates:
                params_buf.append([candidate, x, y, metric_f, itcf])

            params_sl.append(slice(len(params_buf)))
            params += params_buf
        candidates_d = [0]*dim
        for i in [j for j in xrange(dim) if improving_d[j]]:  # MA
            candidates_d[i] = ersatz.elongate(
                c_winner_d[i]['config'], full_dic_d)

            params_buf = []
            for candidate in candidates_d[i]:
                params_buf.append([candidate, x, yd[i], metricd_f, itcf])
            params_sl.append(slice(len(params), len(params)+len(params_buf)))
            params += params_buf

        # pool
        if len(params) > 0:
            results = p.map(try_configuration_p, params)
        else:
            break

        # collect data
        k = 0
        if improving:  # muscle length
            if return_progress > 1:
                dump.append([[i_d[0] for i_d in params[params_sl[0]]],
                             deepcopy(results[params_sl[0]])])
            improving, c_winner, cl_winner = _io_p2(
                params[params_sl[0]], results[params_sl[0]], c_winner,
                min_metr, min_metr_sdif)
            if not improving:
                js_improving = True
            k += 1

        for i in [j for j in xrange(dim) if improving_d[j]]:
            if return_progress > 1:
                dump_m[i].append([[i_d[0] for i_d in params[params_sl[k]]],
                                  deepcopy(results[params_sl[k]])])
            improving_d[i], c_winner_d[i], cl_winner_d[i] = _io_p2(
                params[params_sl[k]], results[params_sl[k]],
                c_winner_d[i], min_metr, min_metr_sdif)
            if not improving_d[i]:
                js_improving_d[i] = True
            k += 1

        # run pool for second-term expansion - try to expand just stopped
        # improving polynomials
        for _ in xrange(look_ahead):
            if not js_improving and not any(js_improving_d):
                break

            params = []
            params_sl = []
            if js_improving:  # muscle length
                candidates2 = []
                for candidate in candidates:
                    for can in ersatz.elongate(candidate, full_dic):
                        if can not in candidates2:
                            candidates2.append(can)
                candidates = candidates2

                params_buf = []
                for candidate in candidates:
                    params_buf.append([candidate, x, y, metric_f, itcf])

                params_sl.append(slice(len(params_buf)))
                params += params_buf
            for i in [j for j in xrange(dim) if js_improving_d[j]]:  # MA
                candidates2 = []
                for candidate in candidates_d[i]:
                    for can in ersatz.elongate(candidate, full_dic_d):
                        if can not in candidates2:
                            candidates2.append(can)
                candidates_d[i] = candidates2

                params_buf = []
                for candidate in candidates_d[i]:
                    params_buf.append([candidate, x, yd[i], metricd_f, itcf])
                params_sl.append(slice(len(params), len(params)+len(params_buf)))
                params += params_buf

            # pool
            if len(params) > 0:
                results = p.map(try_configuration_p, params)
            else:
                break

            # collect data
            k = 0
            if js_improving:  # muscle length
                if return_progress > 1:
                    dump[-1][0] += [i_d[0] for i_d in params[params_sl[0]]]
                    dump[-1][1] += deepcopy(results[params_sl[0]])
                improving, c_winner, cl_winner = _io_p2(
                    params[params_sl[0]], results[params_sl[0]], c_winner,
                    min_metr, min_metr_sdif)
                if improving:
                    js_improving = False
                k += 1

            for i in [j for j in xrange(dim) if js_improving_d[j]]:
                if return_progress > 1:
                    dump_m[i][-1][0] += [i_d[0] for i_d in params[params_sl[k]]]
                    dump_m[i][-1][1] += deepcopy(results[params_sl[k]])
                improving_d[i], c_winner_d[i], cl_winner_d[i] = _io_p2(
                    params[params_sl[k]], results[params_sl[k]],
                    c_winner_d[i], min_metr, min_metr_sdif)
                if improving_d[i]:
                    js_improving_d[i] = False
                k += 1

        # Choose a next solution even if it has higher ITC
        if min_metr is not None:
            if not improving:
                if (c_winner['metric'] > min_metr and
                        c_winner['metric'] - cl_winner['metric'] >
                        min_metr_sdif):
                    c_winner = cl_winner
                    if c_winner['metric'] < min_metr:
                        improving = False
                    else:
                        improving = True

            for i in [j for j in xrange(dim) if not improving_d[j]]:
                if (c_winner_d[i]['metric'] > min_metr and
                        c_winner_d[i]['metric'] - cl_winner_d[i]['metric'] >
                        min_metr_sdif):
                    c_winner_d[i] = cl_winner_d[i]
                    if c_winner_d[i]['metric'] < min_metr:
                        improving_d[i] = False
                    else:
                        improving_d[i] = True

        # Integrate and differentiate
        cwds = [ersatz.Ersatz(dim, cwd['config'])
                for cwd in c_winner_d]
        cw = ersatz.Ersatz(dim, c_winner['config'])

        for i, cwd in enumerate(cwds):
            cw |= cwd.integral(i)
        cwds = cw.differential()

        # calculate the vals
        c_winner_d = [
            try_configuration(cwd.config_dic, x, yd[i], metricd_f)
            for i, cwd in enumerate(cwds)]
        c_winner = try_configuration(cw.config_dic, x, y, metric_f)

        if return_progress == 1:
            dump.append(c_winner)
            for i in xrange(dim):
                dump_m[i].append(c_winner_d[i])

        if return_progress > 2:
            dump_l.append(c_winner)

        if verbose and (improving or True in improving_d):
            print ('{}Using {} to fit. Metric: {}, ITC: {},'
                   ' improving L: {}, improving Ms: {}').format(
                verbose_prefix, ersatz.dic2str(c_winner['config']),
                c_winner['metric'], c_winner['ITC'],
                improving, improving_d)

    p.close()
    if verbose:
        print '{}Finished fitting with ersatz {}. Metric: {}, ITC: {}'.format(
            verbose_prefix, ersatz.dic2str(c_winner['config']),
            c_winner['metric'], c_winner['ITC'])

    if isinstance(c_winner['residuals'], (list, tuple, numpy.ndarray)):
        c_winner['residuals'] = numpy.inf
    for i in xrange(dim):
        if isinstance(c_winner_d[i]['residuals'], (list, tuple, numpy.ndarray)):
            c_winner_d[i]['residuals'] = numpy.inf

    if return_progress > 2:
        return (c_winner, c_winner_d, dump, dump_m, dump_l)

    if return_progress:
        return (c_winner, c_winner_d, dump, dump_m)

    return (c_winner, c_winner_d)

