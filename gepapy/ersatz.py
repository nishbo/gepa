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
from itertools import izip
from itertools import combinations_with_replacement as cwr
from re import findall
from copy import deepcopy
from operator import mul
import numpy


def key_sort(keys):
    """Sorts keys in a common usage form. a, M

    To keep consistency between different places that access parameters"""
    answ = []
    if 'a' in keys:
        answ.append('a')
        keys.pop(keys.index('a'))
    if 'M' in keys:
        answ.append('M')
        keys.pop(keys.index('M'))
    return answ + keys


def config_list_sort_key(x):
    """Measures elements inside a list for a sort in a nice fashion

    For example, use:
        sorted(config, key=config_list_sort_key(x))
    """
    return [len(x)] + list(x)


def full_dic_from_dim(dim, rho):
    """Returns a configuration dictionary that contains all possible params
    for a given dimension and power of polynomial.

    Mathematical form:
        f(x) = a + \sum_p^\rho \sum_{i_1<=..<=i_p}^d M_{p,i_1,..,i_p}
                   \prod_j^p x_{i_j}
        where ommitted, assume summation from 1

    Parameters:
        dim > 0: dimension of the function
        rho > 0: maximum polynomial power allowed
    Returns a dictionary with elements:
        a: constant bias
        M: polynomial representation
            (i_1,.., i_p): for M_{i_1,..,i_p} * \prod_j^p x_{i_j}
                where p \in [0 \rho]
    """
    answ = {'a': None}

    if rho > 0:
        answ['M'] = []
        for i in xrange(rho):
            answ['M'] += [list(j) for j in cwr(range(dim), i+1)]

    return answ


def parameter_vector_base(enc_dim, enc_rho):
    enc_fd = full_dic_from_dim(enc_dim, enc_rho)
    spv = sorted(Ersatz(
        enc_dim, enc_fd,
        point=[0]*length(enc_fd)).similar_parts(agnostic=True))
    del spv[-1]  # removes intercept a
    return spv


def make_A(x, config_dic):
    """Constructs A for y=Ap based on configuration"""
    A = []
    if not isinstance(x, numpy.ndarray):
        x = numpy.array(x)
    if 'a' in config_dic.keys():
        A += [numpy.ones(len(x))]

    if 'M' in config_dic.keys():
        for mi in config_dic['M']:
            # Ac = numpy.ones(len(x))
            # for mii in mi:
            #     Ac *= xn[:, mii]
            #     # Ac = [Ac[j] * x[j][mii] for j in xrange(len(x))]

            # A += [Ac]

            A += [numpy.prod(x[:, mi], axis=1)]

    if len(A) == 0:
        A = [numpy.zeros(len(x))]

    A = numpy.vstack(A).T
    return A


def dic2str(config_dic):
    """Transforms dicionary representation of a configuration into a string
    representation.
    """
    answ = ''
    for key in key_sort(config_dic.keys()):
        if key == 'a':
            answ += 'a'
        if key == 'M':
            answ += 'M['
            for mi in config_dic['M']:
                answ += '(' + ','.join(str(i) for i in mi) + ')'
            answ += ']'
    return answ


def str2dic(s):
    """Transforms string representation of a configuration into a dictionary
    representation.
    """
    answ = {}
    for k in findall('a|M\[[0-9,\(\)\s]*\]', s):
        if k == 'a':
            answ['a'] = None
        elif k[0] == 'M':
            kint = k[2:-1]  # remove M[]
            answ['M'] = []
            for el in findall('\([0-9,]*\)', kint):
                elint = el[1:-1]  # remove '(' '))'
                answ['M'] += [[int(i.strip()) for i in elint.split(',')]]
    return answ


def length(config_dic):
    """Calculates length (amount of parameters) of the given configuration"""
    answ = 0
    for key in config_dic.keys():
        if key == 'a':
            answ += 1
        else:
            answ += len(config_dic[key])
    return answ


def elongate(current_config_dic, full_config_dic):
    """Returns a list of elongated candidate dics towards full_config_dic

    Both input and output configurations in list form."""
    candidates = []

    for k in full_config_dic.keys():
        if k == 'a':
            if 'a' not in current_config_dic.keys():
                c = deepcopy(current_config_dic)
                c['a'] = None
                candidates.append(c)
            continue

        if k not in current_config_dic.keys():
            for el in full_config_dic[k]:
                c = deepcopy(current_config_dic)
                c[k] = [el]
                candidates.append(c)
        else:
            for el in full_config_dic[k]:
                if el not in current_config_dic[k]:
                    c = deepcopy(current_config_dic)
                    c[k].append(el)
                    c[k].sort(key=config_list_sort_key)
                    candidates.append(c)

    return candidates


def get_similarity_parts(dic, agnostic, id_dofs, count_reps):
    ldoc = set()
    for key in key_sort(dic.keys()):
        if key == 'a':
            ldoc.add('a')
            continue

        if key == 'M':
            for el in dic['M']:
                buf = {}  # dim: power
                for e in el:
                    if e not in buf.keys():
                        buf[e] = 0
                    buf[e] += 1

                if agnostic:
                    if count_reps:
                        pows = str(sorted(i[1] for i in buf.items()))
                    else:
                        pows = str(sorted(set(i[1] for i in buf.items())))
                    ldoc.add(pows)
                elif id_dofs is not None:
                    strctr = sorted([id_dofs[i[0]], i[1]] for i in buf.items())
                    strctr = str(strctr)
                    ldoc.add(strctr)
                else:
                    strctr = sorted(buf.items())
                    strctr = str(strctr)
                    ldoc.add(strctr)
    return ldoc


def join(ers):
    """Concatenates and array of Ersatz's
    """
    rer = ers[0]
    for i in ers[1:]:
        rer |= i
    return rer


# Callable class
class Ersatz(object):
    """Uses dictionary of parameters from the fit to create a function"""
    def __init__(self, dim, config, point=None, sdic=None):
        self.dim = dim
        if isinstance(config, str):
            self.config_dic = str2dic(config)
        else:
            self.config_dic = config
        self.point_set = False
        if point is not None:
            self.set(point)
        if sdic is not None:
            self.from_sdic(sdic)

    def sorted_keys(self):
        return key_sort(self.config_dic.keys())

    def set(self, point):
        """set values for function evaluation based on dic"""
        shift = 0
        for key in self.sorted_keys():
            if key == 'a':
                self.a = point[shift]
                shift += 1
                continue

            setattr(self, key, point[shift:shift+len(self.config_dic[key])])
        self.point_set = True

    def get(self):
        """get a point (based on a dic)"""
        point = []
        for key in self.sorted_keys():
            if key == 'a':
                point.append(a)
                continue

            point += getattr(self, key)
        return point

    def from_sdic(self, sdic):
        for term in sdic.keys():
            if term == 'a':
                self.a = sdic['a']
                self.config_dic['a'] = True
                continue

            termd = str2dic(term)
            letter = termd.keys()[0]
            conf = termd[letter][0]
            val = sdic[term]
            if letter not in self.config_dic.keys():
                self.config_dic[letter] = []
                setattr(self, letter, [])
            self.config_dic[letter].append(conf)
            getattr(self, letter).append(val)
        self.point_set = True

        # sort the polynomial terms
        for key in self.config_dic.keys():
            if key == 'a':
                continue
            self.config_dic[key], buf = (
                list(t) for t in zip(*sorted(
                    zip(self.config_dic[key], getattr(self, key)),
                    key=lambda l: config_list_sort_key(l[0]))))
            setattr(self, key, buf)

    def elongate(self, full_config):
        return elongate(self.config_dic, full_config)

    def partial_differential(self, dof):
        """Returns Erzatz representing differential with appropriate point"""
        if not self.point_set:
            # for analytical abstract differentiation
            self.set([0.]*len(self))
            self.point_set = False

        res_sdic = {}  # resulting dic separated into terms together with its
                       # point values
        for key in self.sorted_keys():
            if key == 'a':
                continue

            if key == 'M':
                for mi, mv in izip(self.config_dic['M'], self.M):
                    if dof in mi:
                        cp = sum(1 for mii in mi if mii == dof) # current power
                        rmv = cp*mv
                        rmi = list(mi)
                        rmi.pop(rmi.index(dof))
                        if len(rmi) == 0:
                            rmi = 'a'
                        else:
                            rmi = dic2str({'M': [rmi]})
                        if rmi not in res_sdic.keys():
                            res_sdic[rmi] = 0.
                        res_sdic[rmi] += rmv
                continue

        # Create resulting erzatz
        # print res_sdic
        rer = Ersatz(self.dim, {}, sdic=res_sdic)
        if not self.point_set:
            rer.point_set = False

        return rer

    def differential(self):
        return [self.partial_differential(i) for i in xrange(self.dim)]

    def integral(self, dof):
        """Returns Erzatz representing integral with appropriate point

        Constant (a) is returned as ZERO
        """
        if not self.point_set:
            # for analytical abstract integration
            self.set([0.]*len(self))
            self.point_set = False

        res_sdic = {'a': 0.}  # resulting dic separated into terms together
                              # with its point values
        for key in self.sorted_keys():
            if key == 'a':
                term = dic2str({'M': [[dof]]})
                if term not in res_sdic.keys():
                    res_sdic[term] = 0.
                res_sdic[term] += self.a
                continue

            if key == 'M':
                for mi, mv in izip(self.config_dic['M'], self.M):
                    cp = sum(1 for mii in mi if mi == dof)
                    rmv = mv / (cp+1)
                    rmi = sorted([dof] + mi)
                    rmi = dic2str({'M': [rmi]})
                    if rmi not in res_sdic.keys():
                        res_sdic[rmi] = 0.
                    res_sdic[rmi] += rmv

        # Create resulting erzatz
        # print res_sdic
        rer = Ersatz(self.dim, {}, sdic=res_sdic)
        if not self.point_set:
            rer.point_set = False

        return rer

    def concatenate(self, er2):
        """Concatenates polynomial structures of two Ersatz's

        Produces an Ersatz that contains terms existing in any of the
        polynomials. If point was not set in any of the provided Ersatz's, the
        resulting Ersatz will also have no point set.
        """
        if not self.dim == er2.dim:
            raise ValueError(
                'Dimensions of two Ersatz have to be equal for concatenation.')
        point_set = True
        if not self.point_set:
            point_set = False
            self.set([0.]*len(self))
            self.point_set = False
        if not er2.point_set:
            point_set = False
            er2.set([0.]*len(er2))
            er2.point_set = False

        er1k = self.config_dic.keys()
        er2k = er2.config_dic.keys()
        rer = Ersatz(self.dim, {})
        for key in key_sort(list(set(er1k + er2k))):
            if key in er1k and key not in er2k:
                rer.config_dic[key] = self.config_dic[key]
                setattr(rer, key, getattr(self, key))
            elif key not in er1k and key in er2k:
                rer.config_dic[key] = er2.config_dic[key]
                setattr(rer, key, getattr(er2, key))
            else:
                if key == 'a':
                    rer.config_dic['a'] = True
                    rer.a = self.a + er2.a
                    continue

                rer.config_dic[key], buf = (
                    list(t) for t in zip(*sorted(
                        zip(list(self.config_dic[key]) +
                            list(er2.config_dic[key]),
                            list(getattr(self, key)) +
                            list(getattr(er2, key))),
                        key=lambda l: config_list_sort_key(l[0]))))
                setattr(rer, key, buf)
                for i in xrange(len(rer.config_dic[key])-1, 0, -1):
                    if rer.config_dic[key][i] == rer.config_dic[key][i-1]:
                        rer.config_dic[key].pop(i)
                        getattr(rer, key)[i-1] += getattr(rer, key)[i]
                        getattr(rer, key).pop(i)

        rer.point_set = point_set

        return rer

    def subtract(self, er2):
        """Substracts polynomial structures of two Ersatz's

        Produces an Ersatz that contains terms existing in this but lacking in
        the other. If point was not set in any of the provided Ersatz's, the
        resulting Ersatz will also have no point set.
        """
        if not self.dim == er2.dim:
            raise ValueError(
                'Dimensions of two Ersatz have to be equal for subtraction.')
        point_set = True
        if not self.point_set:
            point_set = False
            self.set([0.]*len(self))
            self.point_set = False
        if not er2.point_set:
            point_set = False
            er2.set([0.]*len(er2))
            er2.point_set = False

        er1k = self.config_dic.keys()
        er2k = er2.config_dic.keys()
        rer = Ersatz(self.dim, {})
        for key in key_sort(er1k):
            if key not in er2k:
                rer.config_dic[key] = deepcopy(self.config_dic[key])
                setattr(rer, key, getattr(self, key))
            else:
                if key == 'a':
                    continue

                setattr(rer, key, [])
                rer.config_dic[key] = []
                for el, v in izip(self.config_dic[key], getattr(self, key)):
                    if el not in er2.config_dic[key]:
                        rer.config_dic[key].append(deepcopy(el))
                        getattr(rer, key).append(v)
                if len(rer.config_dic[key]) == 0:
                    del rer.config_dic[key]

        rer.point_set = point_set

        return rer

    def similar_parts(self, agnostic=False, count_reps=False, id_dofs=None,
                      condim=False):
        """Returns a dictionary for comparison with other Ersatzs"""
        ldoc = {}
        for key in self.sorted_keys():
            if key == 'a':
                if 'a' not in ldoc.keys():
                    ldoc['a'] = 0.
                ldoc['a'] += abs(self.a)
                continue

            if key == 'M':
                for el, val in izip(self.config_dic['M'], self.M):
                    buf = {}  # dim: power
                    for e in el:
                        if e not in buf.keys():
                            buf[e] = 0
                        buf[e] += 1

                    if agnostic:
                        if condim:
                            strctr = str(reversed(i[1] for i in buf.items())[:condim])
                        else:
                            strctr = str(sorted(i[1] for i in buf.items()))
                        # else:
                        #     strctr = str(sorted(set(i[1]
                        #                             for i in buf.items())))
                    elif id_dofs is not None:
                        strctr = str(sorted([id_dofs[i[0]], i[1]]
                                            for i in buf.items()))
                    else:
                        strctr = str(sorted(buf.items()))

                    if strctr not in ldoc.keys():
                        ldoc[strctr] = 0.
                    ldoc[strctr] += abs(val)
        return ldoc

    def similarity_index(self, erb, agnostic=False, count_reps=False,
                         id_dofsa=None, id_dofsb=None, use_point=False,
                         use_point_difference=False, normalize=True,
                         vsp=False, condim=False):
        """Calculates similarity between structures of two polynomials

        If agnostic is specified, id_dofsa and id_dofsb are ignored
        id_dofsa and id_dofsb are needed when polynomials are defined over
        different groups of variables.
        If count_reps is specified, combinations of powers are allowed in
        comparison

        """
        ldoca = self.similar_parts(agnostic=agnostic, id_dofs=id_dofsa,
                                   count_reps=count_reps, condim=condim)
        ldocb = erb.similar_parts(agnostic=agnostic, id_dofs=id_dofsb,
                                  count_reps=count_reps, condim=condim)

        Nc = 0.
        dc = 0.
        Nanc = 0.
        danc = 0.
        vsp_val = 0.
        if vsp:
            # if 'a' not in ldoca.keys():
            #     raise Warning('Lacking intercept in comparison')
            # else:
            if normalize:
                # Normalize so the vector in the space has modulus 1
                amax = sum(i[1]**2 for i in ldoca.items())**0.5
                bmax = sum(i[1]**2 for i in ldocb.items())**0.5
            for k in ldoca.keys():
                if normalize:
                    ldoca[k] /= amax
                ldoca[k] = abs(ldoca[k])
            if normalize:
                del ldoca['a']
            # if 'a' not in ldocb.keys():
            #     raise Warning('Lacking intercept in comparison')
            # else:
            for k in ldocb.keys():
                if normalize:
                    ldocb[k] /= bmax
                ldocb[k] = abs(ldocb[k])
            if normalize:
                del ldocb['a']

        for ka, ea in ldoca.iteritems():
            danc += abs(ea)
            if ka in ldocb.keys():
                vsp_val += (ea - ldocb[ka])**2
                if use_point:
                    Nc += (abs(ea) + abs(ldocb[ka])) / 2.
                elif use_point_difference:
                    dc += abs(ea - ldocb[ka])
                else:
                    Nc += 1.
            else:
                vsp_val += (ea)**2
                if use_point:
                    Nanc += abs(ea)
                else:
                    Nanc += 1.

        Nbnc = 0.
        dbnc = 0.
        for kb, eb in ldocb.iteritems():
            dbnc += abs(eb)
            if kb not in ldoca.keys():
                vsp_val += (eb)**2
                if use_point:
                    Nbnc += abs(eb)
                else:
                    Nbnc += 1.

        if use_point_difference:
            if vsp:
                return (vsp_val)**0.5
            elif normalize:
                return dc / (danc + dbnc)
            else:
                return dc
        else:
            if normalize:
                return Nc / (Nc + Nanc + Nbnc)
            else:
                return Nc

    def structural_difference(self, erb, agnostic=False, count_reps=False,
                              id_dofsa=None, id_dofsb=None, use_point=False,
                              use_point_difference=False, normalize=True,
                              vsp=False, condim=False):
        return 1. - self.similarity_index(
            erb, agnostic=agnostic, count_reps=count_reps,
            id_dofsa=id_dofsa, id_dofsb=id_dofsb, use_point=use_point,
            use_point_difference=use_point_difference, normalize=normalize,
            vsp=vsp, condim=condim)

    def parameter_vector(self, spv):
        """spv - supporting parameter vector"""
        vec = [0.]*len(spv)
        ldoc = self.similar_parts(agnostic=True)

        # remove intercept
        if 'a' in ldoc.keys():
            del ldoc['a']

        for k, v in ldoc.iteritems():
            if k in spv:
                vec[spv.index(k)] = v
            else:
                print 'Key {} is lacking in base vector {}'.format(k, spv)
                raise Warning('Key {} is lacking in base vector {}'.format(
                    k, spv))
        nrm = sum(i**2 for i in vec)**0.5
        vec = [i/nrm for i in vec]
        return vec


    def complexity(self, rho):
        """Calculates the complexity of approximation in terms of parameter
        space portion occupied

        """
        return float(len(self)) / length(full_dic_from_dim(self.dim, rho))

    def c_str(self, xnames):
        """Returns a string with code that can be executed in C-code

        xnames: list of strings to use as placeholders for each dimention
        """
        if not self.point_set:
            raise ValueError('Point not set')
        if not len(xnames) == self.dim:
            print xnames, self.dim
            raise ValueError('Wrong dimensionality of placeholder names')

        parts = []
        for key in self.sorted_keys():
            if key == 'a':
                parts.append('{}'.format(self.a))
            elif key == 'M':
                parts.append(' + '.join('{} * {}'.format(
                        mv, ' * '.join(xnames[i] for i in mi))
                    for mi, mv in izip(self.config_dic['M'], self.M)))
        return ' + '.join(parts)

    def __or__(self, other):
        return self.concatenate(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __call__(self, x):
        """M: [k, l, i, j]"""
        if not self.point_set:
            raise ValueError('Point not set')

        if isinstance(x[0], (int, float)):
            answ = 0.
            if 'a' in self.config_dic:
                answ += self.a
            if 'M' in self.config_dic.keys():
                answ += sum(m * reduce(mul, [x[mii] for mii in mi])
                            for m, mi in izip(self.M, self.config_dic['M']))
        else:
            answ = numpy.zeros(len(x))
            if not isinstance(x, numpy.ndarray):
                x = numpy.array(x)
            if 'a' in self.config_dic:
                answ += self.a
            if 'M' in self.config_dic.keys():
                for mv, mi in izip(self.M, self.config_dic['M']):
                    answ += mv * numpy.prod(x[:, mi], axis=1)
                # answ += sum(m * reduce(mul, [x[mii] for mii in mi])
                #             for m, mi in izip(self.M, self.config_dic['M']))

        return answ

    def __len__(self):
        return length(self.config_dic)

    def __repr__(self):
        return 'gepapy.ersatz.Ersatz({0}, {1}{2})'.format(
            self.dim, self.config_dic,
            ', {}'.format(self.get()) if self.point_set else '')

    def __str__(self):
        s = 'f('
        s += ','.join('x{}'.format(i) for i in xrange(self.dim))
        s += ') = '
        parts = []
        for key in self.sorted_keys():
            if key == 'a':
                if self.point_set:
                    parts.append('{}'.format(self.a))
                else:
                    parts.append('?')
            if key == 'M':
                if self.point_set:
                    parts.append(' + '.join('{}*{}'.format(
                            mv, '*'.join('x{}'.format(i) for i in mi))
                        for mi, mv in izip(self.config_dic['M'], self.M)))
                else:
                    parts.append(' + '.join('?*{}'.format(
                            '*'.join('x{}'.format(i) for i in mi))
                        for mi in self.config_dic['M']))
        s += ' + '.join(parts)

        if len(self.config_dic.keys()) == 0:
            s += '0'
        s += '.'
        return s


def _difint_test():
    # test for differential:
    dim = 3
    rho = 3
    e = Ersatz(dim, full_dic_from_dim(dim, rho))
    e.set([i+1 for i in xrange(len(e))])

    difs = []
    ints = []
    cints = []
    print '\t\tWith point set:'
    print 'Original F:             {}'.format(e)
    for i in xrange(dim):
        difs.append(e.partial_differential(i))
        ints.append(difs[-1].integral(i))
        print 'Differential dF/dx{0}:    {1}'.format(i, difs[-1])
        print 'Integral of dF/dx{0} dx{0}: {1}'.format(i, ints[-1])
    print 'Concatenated integrals: {}'.format(join(ints))
    for i, j in [(i, j) for i in xrange(dim) for j in xrange(dim) if i!=j]:
        cints.append(difs[i].integral(j))
        print 'Integral of dF/dx{0} dx{1}: {2}'.format(i, j, cints[-1])
    print 'Concatenated cross-integrals: {}'.format(join(cints))

    print '\n\t\tWithout point set:'
    e.point_set = False
    difs = []
    ints = []
    cints = []
    print 'Original F:             {}'.format(e)
    for i in xrange(dim):
        difs.append(e.partial_differential(i))
        ints.append(difs[-1].integral(i))
        print 'Differential dF/dx{0}:    {1}'.format(i, difs[-1])
        print 'Integral of dF/dx{0} dx{0}: {1}'.format(i, ints[-1])
    print 'Concatenated integrals: {}'.format(join(ints))
    for i, j in [(i, j) for i in xrange(dim) for j in xrange(dim) if i!=j]:
        cints.append(difs[i].integral(j))
        print 'Integral of dF/dx{0} dx{1}: {2}'.format(i, j, cints[-1])
    print 'Concatenated cross-integrals: {}'.format(join(cints))


def main():
    # _full_dic_test()
    # print
    # print
    # _printrepr_test()
    # print
    # print
    _difint_test()


if __name__ == '__main__':
    main()
