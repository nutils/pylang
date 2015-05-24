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


from . import core


def for_loop_helper(extended_block, dtype, *ranges):

    return _for_loop_helper(extended_block, dtype, ranges, ())

def _for_loop_helper(extended_block, dtype, ranges, outer_loop_vars):

    assert len(ranges) > 0
    if len(ranges) == 1:
        return _for_loop_helper1(extended_block, dtype, ranges[0],
            outer_loop_vars)
    else:
        this_entry, this_exit, this_loop_var = _for_loop_helper1(
            extended_block, dtype, ranges[0], outer_loop_vars)
        inner_entry, inner_exit, *inner_loop_vars = \
            _for_loop_helper(this_entry, dtype, ranges[1:],
                tuple(outer_loop_vars)+(this_loop_var,))
        this_entry.branch(this_exit)
        return [inner_entry, inner_exit, this_loop_var]+inner_loop_vars

def _for_loop_helper1(extended_block, dtype, range_args, outer_loop_vars):

    if callable(range_args):
        range_args = range_args(*outer_loop_vars)
    if not isinstance(range_args, (list, tuple)):
        range_args = [range_args]
    else:
        range_args = list(range_args)
    if len(range_args) == 1:
        range_args.insert(0, 0)
    if len(range_args) == 2:
        range_args.append(1)
    if len(range_args) == 3:
        range_args.append(False)
    start, stop, step, at_least_once = range_args

    eb_loop_start, eb_loop_body, eb_loop_end, eb_loop_exit = \
        extended_block.append_extended_blocks(4)

    loop_var = eb_loop_start.add_phi_node(dtype)
    loop_var.set_value(extended_block.eval(dtype(start)), extended_block._tail)
    extended_block._tail.branch1(eb_loop_start._head)

    eb_loop_start.branch([core.lt(loop_var, stop), eb_loop_body], eb_loop_exit)

    loop_var.set_value(eb_loop_end.eval(loop_var+step), eb_loop_end)
    eb_loop_end.branch(eb_loop_start)

    # TODO: use at_least_once

    extended_block._tail = eb_loop_exit._tail

    return eb_loop_body, eb_loop_end, loop_var


class _attributes:

    __slots__ = ['_attributes']

    def __new__(cls, **attributes):

        self = super().__new__(cls)
        self._attributes = attributes

    def __getattr__(self, attr):

        try:
            return self._attributes[attr]
        except KeyError:
            raise AttributeError


def link_libc(module):

    class libc:

        # TODO: depends on the target platform
        size_t = core.uint64_t
        ssize_t = core.int64_t
        intptr_t = core.int64_t
        uintptr_t = core.uint64_t

        malloc = module.declare_function(
            'malloc',
            [core.int8_t.pointer, 'noalias'], size_t,
            function_attributes=['nounwind'])
        alligned_alloc = module.declare_function(
            'alligned_alloc',
            [core.int8_t.pointer, 'noalias'], size_t, size_t,
            function_attributes=['nounwind'])
        free = module.declare_function(
            'free',
            core.void_t, [core.int8_t.pointer, 'nocapture'],
            function_attributes=['nounwind'])
        printf = module.declare_function(
            'printf',
            core.int32_t, [core.int8_t.pointer, 'nocapture', 'readonly'], ...)

    return libc

# vim: ts=4:sts=4:sw=4:et
