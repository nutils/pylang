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


import subprocess
import tempfile
import os
import cffi
import collections
import itertools

from . import core


def _equivalent_c_dtype(dtype, typedefs):

    if isinstance(dtype, core.VoidType):
        return 'void'
    elif isinstance(dtype, core.PointerType):
        return _equivalent_c_dtype(dtype.reference_dtype, typedefs)+'*'
    elif isinstance(dtype, core.SignedIntegerType):
        return 'int{}_t'.format(dtype.bits)
    elif isinstance(dtype, core.UnsignedIntegerType):
        return 'uint{}_t'.format(dtype.bits)
    elif isinstance(dtype, core.FloatType):
        return dtype._llvm_id
    elif isinstance(dtype, core.FunctionType):
        if dtype in typedefs:
            return typedefs[dtype][0]
        else:
            name = '__pylang_typedef_{:04d}'.format(len(typedefs))
            arg_dtypes = [
                _equivalent_c_dtype(d, typedefs)
                for d in dtype._arguments_dtypes]
            name = '__pylang_typedef_{:04d}'.format(len(typedefs))
            typedefs[dtype] = name, \
                'typedef {} {}({});'.format(
                    _equivalent_c_dtype(dtype._return_dtype, typedefs),
                    name,
                    ', '.join(arg_dtypes))
            return name
    elif isinstance(dtype, core.StructureType):
        if dtype.packed:
            raise NotImplementedError
        else:
            if dtype in typedefs:
                return typedefs[dtype][0]
            else:
                sub_dtypes = [
                    '{} element_{:04d};'.format(
                        _equivalent_c_dtype(d, typedefs),
                        i)
                    for i, d in enumerate(dtype._dtypes)]
                name = '__pylang_typedef_{:04d}'.format(len(typedefs))
                typedefs[dtype] = name, \
                    'typedef struct {{{}}} {};'.format(
                        ' '.join(sub_dtypes), name)
                return name
    else:
        raise NotImplementedError


def compile_and_load(module):

    delete_paths = []
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.so', delete=False)\
                as lib:
            lib_path = lib.name
            delete_paths.append(lib_path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False)\
                as source:
            source_path = source.name
            delete_paths.append(source_path)
            module._generate_ir(source)
        clang_args = ['clang', '-Wall', '-O3', '-shared', '-fPIC',
            '-o', lib_path]
        for ir in module._link_ir:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.ll', delete=False) as extra_file:
                delete_paths.append(extra_file.name)
                extra_file.write(ir)
                clang_args.append(extra_file.name)
        clang_args.append(source_path)
        subprocess.check_call(clang_args)
        ffi = cffi.FFI()
        function_cdefs = []
        typedefs = collections.OrderedDict()
        eq_c_dtype = lambda d: _equivalent_c_dtype(d, typedefs)
        for name, function in module._functions.items():
            dtype = function.dtype.reference_dtype
            function_cdefs.append('{} {}({});'.format(
                eq_c_dtype(dtype._return_dtype),
                name,
                ', '.join(itertools.chain(
                    map(eq_c_dtype, dtype._arguments_dtypes),
                    ('...',) if dtype._variable_arguments else ()))))
        ffi.cdef('\n'.join(itertools.chain(
            (typedef for name, typedef in typedefs.values()),
            function_cdefs)))
        return ffi.dlopen(lib_path)
    finally:
        for path in delete_paths:
            if os.path.exists(path):
                os.remove(path)


# vim: ts=4:sts=4:sw=4:et
