#! /usr/bin/env python3
#
# Copyright (c) 2015 Joost van Zwieten
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import pylang.core, pylang.ee
import operator


def operator_rtzdiv(l, r):

    if isinstance(l, pylang.core.Expression):
        assert isinstance(r, pylang.core.Expression)
        value = pylang.core.rtzdiv._operator_call(l, r)
        if value == NotImplemented:
            raise TypeError
        else:
            return value
    else:
        assert not isinstance(r, pylang.core.Expression)
        sign = 1 if l*r > 0 else -1
        return sign*(abs(l)//abs(r))


def eval_expression(expression):

    module = pylang.core.Module()
    test, entry = module.define_function('test', expression.dtype)
    entry.ret(expression)

    mod = pylang.ee.compile_and_load(module)
    return mod.test()


def compare(dtype, op, *args, tol=None):

    a = op(*args)
    b = eval_expression(op(*map(dtype, args)))
    if tol and a != 0:
        return abs(a-b) < tol*abs(a)
    else:
        return a == b


class TestSignedIntegerArithmetic(unittest.TestCase):

    dtypes = pylang.core.int8_t, pylang.core.int16_t, pylang.core.int32_t, \
        pylang.core.int64_t

    def test_neg(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.neg, 2))

    def test_add(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.add, 2, 3))

    def test_sub(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.sub, 2, 3))

    def test_mul(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.mul, 2, 3))

    def test_truediv(self):

        for dtype in self.dtypes:
            with self.assertRaises(TypeError):
                compare(dtype, operator.truediv, 2, 3)

    def test_floordiv(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.floordiv, 2, 3))
            self.assertTrue(compare(dtype, operator.floordiv, 3, 3))
            self.assertTrue(compare(dtype, operator.floordiv, 7, 3))
            self.assertTrue(compare(dtype, operator.floordiv, -1, 3))
            self.assertTrue(compare(dtype, operator.floordiv, -3, 3))
            self.assertTrue(compare(dtype, operator.floordiv, -1, -3))
            self.assertTrue(compare(dtype, operator.floordiv, -3, -3))
            self.assertTrue(compare(dtype, operator.floordiv, 1, -3))
            self.assertTrue(compare(dtype, operator.floordiv, 3, -3))

    def test_rtzdiv(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator_rtzdiv, 2, 3))
            self.assertTrue(compare(dtype, operator_rtzdiv, 3, 3))
            self.assertTrue(compare(dtype, operator_rtzdiv, 7, 3))
            self.assertTrue(compare(dtype, operator_rtzdiv, -1, 3))
            self.assertTrue(compare(dtype, operator_rtzdiv, -3, 3))
            self.assertTrue(compare(dtype, operator_rtzdiv, -1, -3))
            self.assertTrue(compare(dtype, operator_rtzdiv, -3, -3))
            self.assertTrue(compare(dtype, operator_rtzdiv, 1, -3))
            self.assertTrue(compare(dtype, operator_rtzdiv, 3, -3))


class TestUnsignedIntegerArithmetic(unittest.TestCase):

    dtypes = pylang.core.uint8_t, pylang.core.uint16_t, pylang.core.uint32_t, \
        pylang.core.uint64_t

    def test_neg(self):

        for dtype in self.dtypes:
            with self.assertRaises(TypeError):
                compare(dtype, operator.neg, 2)

    def test_add(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.add, 2, 3))

    def test_sub(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.sub, 3, 2))

    def test_mul(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.mul, 2, 3))

    def test_truediv(self):

        for dtype in self.dtypes:
            with self.assertRaises(TypeError):
                compare(dtype, operator.truediv, 2, 3)

    def test_floordiv(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator.floordiv, 2, 3))
            self.assertTrue(compare(dtype, operator.floordiv, 3, 3))
            self.assertTrue(compare(dtype, operator.floordiv, 7, 3))

    def test_rtzdiv(self):

        for dtype in self.dtypes:
            self.assertTrue(compare(dtype, operator_rtzdiv, 2, 3))
            self.assertTrue(compare(dtype, operator_rtzdiv, 3, 3))
            self.assertTrue(compare(dtype, operator_rtzdiv, 7, 3))


class TestFloatArithmetic(unittest.TestCase):

    dtypes = (pylang.core.float32_t, 2**-23), (pylang.core.float64_t, 2**-52)

    def test_neg(self):

        for dtype, tol in self.dtypes:
            self.assertTrue(compare(dtype, operator.neg, 2, tol=tol))

    def test_add(self):

        for dtype, tol in self.dtypes:
            self.assertTrue(compare(dtype, operator.add, 2, 3, tol=tol))

    def test_sub(self):

        for dtype, tol in self.dtypes:
            self.assertTrue(compare(dtype, operator.sub, 3, 2, tol=tol))

    def test_mul(self):

        for dtype, tol in self.dtypes:
            self.assertTrue(compare(dtype, operator.mul, 2, 3, tol=tol))

    def test_truediv(self):

        for dtype, tol in self.dtypes:
            self.assertTrue(compare(dtype, operator.truediv, 2, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, 3, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, 7, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, -1, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, -3, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, -1, -3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, -3, -3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, 1, -3, tol=tol))
            self.assertTrue(compare(dtype, operator.truediv, 3, -3, tol=tol))

    def test_floordiv(self):

        for dtype, tol in self.dtypes:
            self.assertTrue(compare(dtype, operator.floordiv, 2, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, 3, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, 7, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, -1, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, -3, 3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, -1, -3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, -3, -3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, 1, -3, tol=tol))
            self.assertTrue(compare(dtype, operator.floordiv, 3, -3, tol=tol))

    def test_rtzdiv(self):

        for dtype, tol in self.dtypes:
            self.assertTrue(compare(dtype, operator_rtzdiv, 2, 3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, 3, 3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, 7, 3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, -1, 3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, -3, 3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, -1, -3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, -3, -3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, 1, -3, tol=tol))
            self.assertTrue(compare(dtype, operator_rtzdiv, 3, -3, tol=tol))


# vim: ts=4:sts=4:sw=4:et
