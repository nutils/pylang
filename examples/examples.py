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

import pylang
from pylang.core import *
from pylang import utils, ee, foreign


def compile_and_run(module):

    if '--show-ir' in sys.argv:
        print('-'*80)
        module._generate_ir(sys.stdout)
        print('-'*80)

    mod = ee.compile_and_load(module)
    mod.main()


def test1():

    module = pylang.core.Module()

    main, entry = module.define_function('main', int32_t)
    body, end = entry.append_extended_blocks(2)

    entry.branch(body)

    i = body.add_phi_node(int32_t)
    factorial_previous = body.add_phi_node(int32_t)
    i_inc = body.eval(i+1)
    factorial = body.eval(factorial_previous*i_inc)
    body.branch([lt(i_inc, 5), body], end)
    i.set_value(0, entry)
    i.set_value(i_inc, body)
    factorial_previous.set_value(1, entry)
    factorial_previous.set_value(factorial, body)

    end.call('printf', b'5! = %d\n', factorial)
    end.ret(0)

    compile_and_run(module)


def test2():

    module = Module()

    main, entry = module.define_function('main', int32_t)
    body, exit = entry.append_extended_blocks(2)
    entry.branch(body)

    n=5
    i = body.add_phi_node(int32_t)
    array = DynamicArray.allocate(body, int32_t, n)
    body.assign(array[i], i)

    body.call('printf', b'%d\n', array[i])

    i.set_value(0, entry)
    i.set_value(body.eval(i+1), body)
    body.branch([lt(i+1, n), body], exit)

    array.free(exit)
    exit.ret(int32_t(0))

    compile_and_run(module)


def test3():

    module = Module()

    main, entry = module.define_function('main', int32_t)

    loop_entry, loop_exit, i, j, k = utils.for_loop_helper(entry,
        int32_t, 2, lambda i: i+2, lambda i, j: i+j+2)
    loop_entry.call('printf', b'i = %d, j = %d, k = %d\n', i, j, k)
    loop_entry.branch(loop_exit)

    entry.ret(0)

    compile_and_run(module)


class DynamicArray:

    def __new__(cls, address, element_dtype, shape, strides):

        self = super().__new__(cls)
        self.address = address
        self._element_dtype = element_dtype
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        assert len(self.shape) == len(self.strides)
        return self

    @classmethod
    def allocate(cls, eb, dtype, n_elements):

        if isinstance(n_elements, (list, tuple)):
            shape = tuple(n_elements)
            n_elements = 1
            strides = []
            for l in reversed(shape):
                strides.insert(0, n_elements)
                n_elements *= l
            strides = tuple(strides)
        else:
            shape = n_elements,
            strides = 1,

        sizeof_dtype = eb._module.sizeof(dtype)
        pointer = eb.call('malloc', sizeof_dtype*n_elements)
        pointer = eb.eval(dtype.pointer.bitcast(pointer))
        return cls(pointer, dtype, shape, strides)

    def free(self, eb):

        eb.call('free', int8_t.pointer.bitcast(self.address))

    def __getitem__(self, indices):

        if not isinstance(indices, (tuple, list)):
            indices = indices,
        assert len(indices) == len(self.strides)
        flat_index = sum(i*s for i, s in zip(indices, self.strides))
        return self.address[flat_index]

    def reshape(self, new_shape):

        prod = lambda l: functools.reduce(operator.mul, l, 1)

        new_shape = tuple(new_shape)
        assert prod(new_shape) == prod(self.shape)

        stride = 1
        new_strides = []
        for l in reversed(new_shape):
            new_strides.insert(0, stride)
            stride *= l

        return DynamicArray(self.address, self._element_dtype, shape=new_shape,
            strides=new_strides)


def test4():

    module = Module()

    main, entry = module.define_function('main', int32_t)

    a = DynamicArray.allocate(entry, int32_t, (10, 10))
    b = DynamicArray.allocate(entry, int32_t, (10, 10))
    c = DynamicArray.allocate(entry, int32_t, (10, 10))

    loop_entry, loop_exit, *I = utils.for_loop_helper(entry, int32_t, *a.shape)
    loop_entry.assign(a[I], I[0])
    loop_entry.assign(b[I], I[1])
    loop_entry.branch(loop_exit)

    loop_entry, loop_exit, *I = utils.for_loop_helper(entry, int32_t, *a.shape)
    loop_entry.assign(c[I], a[I]+b[I])
    loop_entry.call('printf', b'c[%d,%d] = %d\n', I[0], I[1], c[I])
    loop_entry.branch(loop_exit)

    a.free(entry)
    b.free(entry)
    c.free(entry)
    entry.ret(0)

    compile_and_run(module)


def test_fib_recursive(n):

    module = Module()

    fib, fib_entry, fib_n = module.define_function('fib', int32_t, int32_t)

    fib_lt3, fib_ge3 = fib_entry.append_extended_blocks(2)
    fib_entry.branch([lt(fib_n, 3), fib_lt3], fib_ge3)
    fib_lt3.ret(1)
    fib_ge3.ret(fib_ge3.call(fib, fib_n-1)+fib_ge3.call(fib, fib_n-2))

    mod = ee.compile_and_load(module)

    for i in range(1, n+1):
        print('fib({}) = {}'.format(i, mod.fib(i)))


def test_fib_loop(n):

    module = Module()

    fib, fib_entry, fib_n = module.define_function('fib', int32_t, int32_t)

    fib_small, fib_body, fib_exit = fib_entry.append_extended_blocks(3)

    fib_entry.branch([lt(fib_n, 3), fib_small], fib_body)
    fib_small.ret(1)

    i = fib_body.add_phi_node(int32_t)
    val_1 = fib_body.add_phi_node(int32_t)
    val_2 = fib_body.add_phi_node(int32_t)
    i.set_value(2, fib_entry)
    i.set_value(fib_body.eval(i+1), fib_body)
    val_1.set_value(1, fib_entry)
    val_1.set_value(val_2, fib_body)
    val_2.set_value(1, fib_entry)
    val_2.set_value(fib_body.eval(val_1+val_2), fib_body)
    fib_body.branch([lt(i, fib_n), fib_body], fib_exit)
    fib_exit.ret(val_2)

    mod = ee.compile_and_load(module)

    for i in range(1, n+1):
        print('fib({}) = {}'.format(i, mod.fib(i)))


def test_fib_cplusplus(n):

    module = Module()

    fib, = foreign.define_cplusplus_functions(
        module,
        [['fib', FunctionType(int32_t, (int32_t,), False)]],
        '''
            #include <stdint.h>
            extern "C" int32_t fib(int32_t n)
            {
                if (n < 3)
                    return 1;
                else
                    return fib(n-2) + fib(n-1);
            }
        ''')

    mod = ee.compile_and_load(module)

    for i in range(1, n+1):
        print('fib({}) = {}'.format(i, mod.fib(i)))


def test_typecast():

    module = Module()

    main, entry = module.define_function('main', int32_t)

    i = int32_t(1)
    f = float64_t(i)
    entry.call('printf', b'%f\n', f)

    f = float64_t(1.)
    i = int32_t(f)
    entry.call('printf', b'%d\n', i)

    entry.ret(0)

    compile_and_run(module)


def test_float_loop():

    module = Module()

    main, entry = module.define_function('main', int32_t)

    loop_entry, loop_exit, i = utils.for_loop_helper(entry,
        float64_t, [0, 10, 0.25])
    loop_entry.call('printf', b'%f\n', i)
    loop_entry.branch(loop_exit)

    entry.ret(0)

    compile_and_run(module)


class FrozenOrderedDict(collections.Sequence, collections.Mapping,
        collections.Hashable):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self._data = collections.OrderedDict(*args, **kwargs)
        return self
    def __hash__(self):
        return hash(tuple(self._data.items()))
    def __eq__(self, other):
        return isinstance(other, FrozenOrderedDict) \
            and self._data == other._data
    def __getitem__(self, key):
        return self._data[key]
    def __len__(self):
        return len(self._data)
    def __iter__(self):
        return iter(self._data)
    def keys(self):
        return self._data.keys()
    def values(self):
        return self._data.values()
    def items(self):
        return self._data.items()


def test_struct():

    S = StructureType(FrozenOrderedDict(
        [['foo', int32_t], ['bar', float64_t]]))

    module = Module()

    main, entry = module.define_function('main', int32_t)

    val = LLVMIdentifier('undef', S)
    val = val.insert({'foo': 1, 'bar': 2})

    entry.call('printf', b'%d %f\n', val['foo'], val['bar'])
    entry.ret(0)

    compile_and_run(module)


def test_allocate_stack():

    S = StructureType(FrozenOrderedDict(
        [['foo', int32_t], ['bar', float64_t]]))
    T = StructureType((int64_t, S))

    module = Module()

    main, entry = module.define_function('main', int32_t)

    var = entry.allocate_stack(T, 3)

    entry.assign(var[0][0], 0)
    entry.assign(var[0][1]['foo'], 1)
    entry.assign(var[0][1]['bar'], 2)

    entry.assign(var[1], {0: 0, 1: {'foo': 1, 'bar': 2}})

    entry.assign(var[2], [0, {'foo': 1, 'bar': 2}])

    entry.call('printf', b'%d %d %f\n', var.content[0], var.content[1]['foo'], var.content[1]['bar'])
    entry.call('printf', b'%d %d %f\n', var[1][0], var[1][1]['foo'], var[1][1]['bar'])
    entry.call('printf', b'%d %d %f\n', var[2][0], var[2][1]['foo'], var[2][1]['bar'])
    entry.ret(0)

    compile_and_run(module)


def test_array():

    module = Module()

    main, entry = module.define_function('main', int32_t)

    var = entry.allocate_stack(ArrayType(int32_t, 3))
    var = var.content

    entry.assign(var, [1, 2, 3])
    entry.call('printf', b'%d %d %d\n', var[0], var[1], var[2])
    entry.ret(0)

    compile_and_run(module)


def test_function_pointer():

    module = Module()

    foo, foo_entry, foo_a = module.define_function('foo', int32_t, int32_t)
    foo_entry.ret(foo_a+1)

    bar, bar_entry, bar_func = module.define_function('bar', void_t,
        foo.dtype)
    bar_entry.call('printf', b'%d\n', bar_entry.call(bar_func, 1))
    bar_entry.ret()

    main, main_entry = module.define_function('main', int32_t)
    main_entry.call(bar, foo)
    main_entry.ret(0)

    compile_and_run(module)


if __name__ == '__main__':

    test1()
    test2()
    test3()
    test4()
    test_fib_recursive(10)
    test_fib_loop(10)
    test_fib_cplusplus(10)
    test_typecast()
    test_float_loop()
#   test_struct()
    test_allocate_stack()
    test_array()
    test_function_pointer()

# vim: ts=4:sts=4:sw=4:et
