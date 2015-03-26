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


def define_cplusplus_functions(module, function_names_dtypes, source, *,
        c_std=None):

    # TODO: verify types of functions in `source` with `function_names_dtypes`

    # compile `source` with clang

    source_path = ir_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cc', delete=False)\
                as source_file:
            source_path = source_file.name
            source_file.write(source)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False)\
                as ir_file:
            ir_path = ir_file.name
        clang_args = ['clang++', '-O3', '-S', '-emit-llvm']
        if c_std:
            clang_args.append('-std={}'.format(c_std))
        clang_args.append('-Wall')
        clang_args.extend(['-o', ir_path, source_path])
        subprocess.check_call(clang_args)
        with open(ir_path, 'r') as ir_file:
            ir = ir_file.read()
    finally:
        if source_path:
            os.remove(source_path)
        if ir_path and os.path.exists(ir_path):
            os.remove(ir_path)

    module._link_ir.append(ir)

    # declare functions in `module`

    function_pointers = []
    for name, dtype in function_names_dtypes:
        # TODO: add return value, argument and function attributes from `ir`
        args = tuple(dtype._parameters)
        if dtype._variable_arguments:
            args += ...,
        function_pointers.append(
            module.declare_function(name, dtype._return_value, *args))
    return tuple(function_pointers)


# vim: ts=4:sts=4:sw=4:et
