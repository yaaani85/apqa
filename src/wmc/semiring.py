"""
problog.evaluator - Commone interface for evaluation
----------------------------------------------------

Provides common interface for evaluation of weighted logic formulas.

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
from __future__ import print_function
import torch
import numpy as np

try:
    from numpy import polynomial

    pn = polynomial.polynomial
except ImportError:
    pn = None


class Semiring(object):
    """Interface for weight manipulation.

    A semiring is a set R equipped with two binary operations '+' and 'x'.

    The semiring can use different representations for internal values and external values.
    For example, the LogProbability semiring uses probabilities [0, 1] as external values and uses \
     the logarithm of these probabilities as internal values.

    Most methods take and return internal values. The execeptions are:

       - value, pos_value, neg_value: transform an external value to an internal value
       - result: transform an internal to an external value
       - result_zero, result_one: return an external value

    """

    def one(self):
        """Returns the identity element of the multiplication."""
        raise NotImplementedError()

    def is_one(self, value):
        """Tests whether the given value is the identity element of the multiplication."""
        return value == self.one

    def zero(self):
        """Returns the identity element of the addition."""
        raise NotImplementedError()

    def is_zero(self, value):
        """Tests whether the given value is the identity element of the addition."""
        return value == self.zero()

    def plus(self, a, b):
        """Computes the addition of the given values."""
        raise NotImplementedError()

    def times(self, a, b):
        """Computes the multiplication of the given values."""
        raise NotImplementedError()

    def negate(self, a):
        """Returns the negation. This operation is optional.
        For example, for probabilities return 1-a.

        :raise OperationNotSupported: if the semiring does not support this operation
        """
        raise Exception("operation not supported")

    def value(self, a):
        """Transform the given external value into an internal value."""
        return a.float()

    def result(self, a, formula=None):
        """Transform the given internal value into an external value."""
        return a

    def normalize(self, a, z):
        """Normalizes the given value with the given normalization constant.

        For example, for probabilities, returns a/z.

        :raise OperationNotSupported: if z is not one and the semiring does not support \
         this operation
        """
        if self.is_one(z):
            return a
        else:
            raise Exception("operation not supported")

    def pos_value(self, a, key=None):
        """Extract the positive internal value for the given external value."""
        return self.value(a, key)

    def neg_value(self, a, key=None):
        """Extract the negative internal value for the given external value."""
        return self.negate(self.value(a, key))

    def result_zero(self):
        """Give the external representation of the identity element of the addition."""
        return self.result(self.zero())

    def result_one(self):
        """Give the external representation of the identity element of the multiplication."""
        return self.result(self.one())

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return False

    def is_nsp(self):
        """Indicates whether this semiring requires solving a neutral sum problem."""
        return False

    def in_domain(self, a):
        """Checks whether the given (internal) value is valid."""
        return True

    def ad_complement(self, ws, key=None):
        s = self.zero()
        for w in ws:
            s = self.plus(s, w)
        return self.negate(s)

    def true(self, key=None):
        """Handle weight for deterministically true."""
        return self.one(), self.zero()

    def false(self, key=None):
        """Handle weight for deterministically false."""
        return self.zero(), self.one()

    def to_evidence(self, pos_weight, neg_weight, sign):
        """
        Converts the pos. and neg. weight (internal repr.) of a literal into the case where the literal is evidence.
        Note that the literal can be a negative atom regardless of the given sign.

        :param pos_weight: The current positive weight of the literal.
        :param neg_weight: The current negative weight of the literal.
        :param sign: Denotes whether the literal or its negation is evidence. sign > 0 denotes the literal is evidence,
            otherwise its negation is evidence. Note: The literal itself can also still be a negative atom.
        :returns: A tuple of the positive and negative weight as if the literal was evidence.
            For example, for probability, returns (self.one(), self.zero()) if sign else (self.zero(), self.one())
        """
        return (self.one(), self.zero()) if sign > 0 else (self.zero(), self.one())

    def ad_negate(self, pos_weight, neg_weight):
        """
        Negation in the context of an annotated disjunction. e.g. in a probabilistic context for 0.2::a ; 0.8::b,
        the negative label for both a and b is 1.0 such that model {a,-b} = 0.2 * 1.0 and {-a,b} = 1.0 * 0.8.
        For a, pos_weight would be 0.2 and neg_weight could be 0.8. The returned value is 1.0.
        :param pos_weight: The current positive weight of the literal (e.g. 0.2 or 0.8). Internal representation.
        :param neg_weight: The current negative weight of the literal (e.g. 0.8 or 0.2). Internal representation.
        :return: neg_weight corrected based on the given pos_weight, given the ad context (e.g. 1.0). Internal
        representation.
        """
        return self.one()


# class SemiringProbability(Semiring):
#     """Implementation of the semiring interface for probabilities."""

#     def one(self):
#         return torch.tensor(1.0)

#     def zero(self):
#         return torch.tensor(0.0)

#     def is_one(self, value):
#         return 1.0 - 1e-12 < value < 1.0 + 1e-12

#     def is_zero(self, value):
#         return -1e-12 < value < 1e-12

#     def plus(self, a, b):
#         return a + b

#     def times(self, a, b):
#         return a * b

#     def negate(self, a):
#         return 1.0 - a

#     def normalize(self, a, z):
#         return a / z

#     def value(self, a, key=None):
#         v = a.float()
#         if 0.0 - 1e-9 <= v <= 1.0 + 1e-9:
#             return v
#         else:
#             raise Exception(
#                 "Not a valid value for this semiring: '%s'" % a, location=a.location
#             )

#     def is_dsp(self):
#         """Indicates whether this semiring requires solving a disjoint sum problem."""
#         return True

#     def in_domain(self, a):
#         return 0.0 - 1e-9 <= a <= 1.0 + 1e-9


# class SemiringLogProbability(SemiringProbability):
#     """Implementation of the semiring interface for probabilities with logspace calculations."""

#     inf, ninf = torch.tensor(float("inf")), torch.tensor(float("-inf"))

#     def one(self):
#         return torch.tensor(0.0)

#     def zero(self):
#         return self.ninf

#     def is_zero(self, value):
#         return value <= -1e100

#     def is_one(self, value):
#         return -1e-12 < value < 1e-12

#     def plus(self, a, b):
#         if a < b:
#             if a == self.ninf:
#                 return b
#             return b + torch.log1p(torch.exp(a - b))
#         else:
#             if b == self.ninf:
#                 return a
#             return a + torch.log1p(torch.exp(b - a))

#     def times(self, a, b):
#         return a + b

#     def negate(self, a):
#         if not self.in_domain(a):
#             raise Exception("Not a valid value for this semiring: '%s'" % a)
#         if a > -1e-10:
#             return self.zero()
#         return torch.log1p(-torch.exp(a))

#     def value(self, a, key=None):
#         v = a.float()
#         if -1e-9 <= v < 1e-9:
#             return self.zero()
#         else:
#             if 0.0 - 1e-9 <= v <= 1.0 + 1e-9:
#                 return torch.log(v)
#             else:
#                 raise Exception(
#                     "Not a valid value for this semiring: '%s'" % a
#                 )

#     def result(self, a, formula=None):
#         return torch.exp(a)

#     def normalize(self, a, z):
#         # Assumes Z is in log
#         return a - z

#     def is_dsp(self):
#         """Indicates whether this semiring requires solving a disjoint sum problem."""
#         return True

#     def in_domain(self, a):
#         return a <= 1e-12


# class SemiringGradient(Semiring):

#     def __init__(self):
#         Semiring.__init__(self)

#     def zero(self):
#         return 0.0, 0.0

#     def one(self):
#         return 1.0, 0.0

#     def plus(self, a, b):
#         return a[0]+b[0], a[1]+b[1]

#     def times(self, a, b):
#         return a[0]*b[0], b[0]*a[1]+a[0]*b[1]

#     def value(self, a, key=None):
#         return float(a), 1.0

#     def negate(self, a):
#         return 1.0-a[0], -1.0*a[1]

#     def is_dsp(self):
#         return True

#     def is_one(self, a):
#         return (1.0 - 1e-12 < a[0] < 1.0 + 1e-12) and (a[1] == 0)

#     def is_zero(self, a):
#         return (-1e-12 < a[0] < 1e-12) and (a[1] == 0)

#     def normalize(self, a, z):
#         diff = (a[1]*z[0]-z[1]*a[0])/(z[0]**2)
#         return a[0]/z[0], diff

class SemiringGradient(Semiring):

    def __init__(self, shape):
        Semiring.__init__(self)
        self.shape = shape

    def zero(self):
        return 0.0, np.zeros(self.shape)

    def one(self):
        return 1.0, np.zeros(self.shape)

    def plus(self, a, b):
        return a[0]+b[0], a[1]+b[1]

    def times(self, a, b):
        return a[0]*b[0], b[0]*a[1]+a[0]*b[1]

    def value(self, a, key):
        v = a,
        diff = np.zeros(self.shape)
        diff[key] = 1.0
        return v[0], diff

    def negate(self, a):
        return 1.0-a[0], -1.0*a[1]

    def is_dsp(self):
        return True

    def is_one(self, a):
        return (1.0 - 1e-12 < a[0] < 1.0 + 1e-12) and (np.count_nonzero(a[1]) == 0)

    def is_zero(self, a):
        return (-1e-12 < a[0] < 1e-12) and (np.count_nonzero(a[1]) == 0)

    def normalize(self, a, z):
        diff = np.zeros(self.shape)
        for i in range(self.shape):
            diff[i] = (a[1][i]*z[0]-z[1][i]*a[0])/(z[0]**2)
        return a[0]/z[0], diff
