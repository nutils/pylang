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


import numbers
import weakref
import functools
import operator
import collections
import itertools


class MultipleDispatchFunction:

    def __init__(self, n_args):

        self._n_args = n_args
        self._implementations = []

    def register(self, func):

        self._implementations.append(func)
        return func

    def _operator_call(self, *args):

        assert len(args) == self._n_args
        for func in self._implementations:
            result = func(*args)
            if result is not NotImplemented:
                return result
        return NotImplemented

    def __call__(self, *args):

        result = self._operator_call(*args)
        if result is NotImplemented:
            raise NotImplementedError
        else:
            return result


class TypeBase:

    # Derived class should implement `__getnewargs__` or both `__hash__` and
    # `__eq__`.

    def __new__(cls, llvm_id):

        self = super().__new__(cls)
        self._llvm_id = llvm_id
        return self

    def __getstate__(self):

        return False

    def __hash__(self):

        return hash((self.__class__.__name__, self.__getnewargs__()))

    def __eq__(self, other):

        return type(self) == type(other) \
            and self.__getnewargs__() == other.__getnewargs__()

    def __call__(self, expression):

        if isinstance(expression, Expression) and expression.dtype == self:
            return expression
        else:
            raise NotImplementedError

    @property
    def pointer(self):

        return PointerType(self)

    def _expression_getattr(self, expression, attr):

        raise AttributeError('expresion with dtype {!r} has no attribute {!r}'.format(self, attr))

    def _expression_getitem(self, expression, index):

        raise TypeError('expression with dtype {!r} does not support indexing.'.format(self))


class VoidType(TypeBase):

    def __new__(cls):

        return super().__new__(cls, 'void')

    def __getnewargs__(self):

        return ()


class FunctionType(TypeBase):

    def __new__(cls, return_dtype, arguments_dtypes, variable_arguments):

        arguments_dtypes = tuple(arguments_dtypes)
        variable_arguments = bool(variable_arguments)

        arg_ids = [dtype._llvm_id for dtype in arguments_dtypes]
        if variable_arguments:
            arg_ids.append('...')
        llvm_id = '{} ({})'.format(return_dtype._llvm_id, ', '.join(arg_ids))

        self = super().__new__(cls, llvm_id)
        self._return_dtype = return_dtype
        self._arguments_dtypes = arguments_dtypes
        self._variable_arguments = variable_arguments
        return self

    def __getnewargs__(self):

        return self._return_dtype, self._arguments_dtypes, \
            self._variable_arguments


class FirstClassType(TypeBase):

    def bitcast(self, value):

        if value.dtype == self:
            return value
        assert isinstance(value.dtype, FirstClassType)
        return RHSExpression(
            self,
            lambda _value: 'bitcast {} to {}'.format(
                _value._llvm_ty_val, self._llvm_id),
            (value,))


class AggregateType(FirstClassType):

    pass

    # should implement
    #   def _assign_helper(self, eb, variable, aggregate_expression):


class PointerType(FirstClassType):

    def __new__(cls, reference_dtype):

        assert reference_dtype not in (VoidType, LabelType)

        # FIXME: determine size of pointer
        self = super().__new__(cls, reference_dtype._llvm_id+'*')
        self.reference_dtype = reference_dtype
        return self

    def __getnewargs__(self):

        return self.reference_dtype,

    def __call__(self, value):

        if self.reference_dtype == int8_t and isinstance(value, bytes):
            # TODO: do we want this behaviour?
            return StringConstant(value)
        else:
            assert isinstance(value, Expression)
            assert value.dtype == self
            return value

    def _expression_getattr(self, expression, attr):

        if attr == 'content':
            return DereferencePointer(expression)
        else:
            return super()._expression_getattr(expression, attr)

    def _expression_getitem(self, expression, index):

        return DereferencePointer(GetElementPointer(expression, index, self))


class VectorType(FirstClassType):

    def __new__(cls, dtype, length):

        self = super().__new__(cls, '<{} x {}>'.format(length, dtype._llvm_id))
        self._element_dtype = dtype
        self._length = length
        return self

    def __getnewargs__(self):

        return self._element_dtype, self._length

    def create(self, elements):

        # TODO: use notation mentioned in
        # http://llvm.org/docs/LangRef.html#complex-constants
        # if all elements are constant
        return LLVMIdentifier('undef', self).replace(elements)

    def _expression_getattr(self, expression, attr):

        if attr == 'replace':
            return functools.partial(self._replace, expression)
        else:
            return super()._expression_getattr(expression, attr)

    def _replace(self, expression, elements):

        if isinstance(elements, collections.Mapping):
            for index, element in elements.items():
                expression = InsertElement(expression, element, index)
        elif isinstance(elements, collections.Sequence):
            index = 0
            for element in elements:
                expression = InsertElement(expression, element, index)
                index += 1
            if index != self._length:
                raise ValueError('length of `elements` does not match with'\
                    ' length of vector')
        else:
            raise ValueError('`elements` should be a mapping or a sequence')
        return expression

    def _expression_getitem(self, expression, index):

        return ExtractElement(expression, index)


class ArrayType(AggregateType):

    def __new__(cls, dtype, length):

        self = super().__new__(cls, '[{} x {}]'.format(length, dtype._llvm_id))
        self._element_dtype = dtype
        self._length = length
        return self

    def __getnewargs__(self):

        return self._element_dtype, self._length

    def _expression_getitem(self, expression, index):

        if isinstance(expression, DereferencePointer):
            return DereferencePointer(GetElementPointer(
                expression.address, index, self._element_dtype.pointer))
        else:
            return ExtractValue(expression, index, self._element_dtype)

    def _assign_helper(self, eb, variable, aggregate_expression):

        if isinstance(aggregate_expression, collections.Mapping):
            for index, value in aggregate_expression.items():
                eb.assign(variable[index], value)
        else:
            assert isinstance(aggregate_expression, collections.Sequence)
            n = 0
            for index, value in enumerate(aggregate_expression):
                eb.assign(variable[index], value)
                n += 1
            assert n == self._length


class LabelType(FirstClassType):

    def __new__(cls):

        return super().__new__(cls, 'label')

    def __getnewargs__(self):

        return ()


#class MetadataType(FirstClassType):
#
#    pass


class StructureType(AggregateType):

    def __new__(cls, dtypes, packed=False):

        assert isinstance(dtypes, collections.Sequence) \
            and isinstance(dtypes, collections.Hashable)

        if isinstance(dtypes, collections.Mapping):
            _dtypes = tuple(dtypes.values())
            _get_index = tuple(dtypes.keys()).index
        else:
            _dtypes = dtypes
            _get_index = lambda x: x

        packed = bool(packed)
        if packed:
            fmt = '<{{{}}}>'
        else:
            fmt = '{{{}}}'
        llvm_id = fmt.format(', '.join(dtype._llvm_id for dtype in _dtypes))

        self = super().__new__(cls, llvm_id)
        self.dtypes = dtypes
        self._dtypes = _dtypes
        self._get_index = _get_index
        self.packed = packed
        return self

    def __getnewargs__(self):

        return self.dtypes, self.packed

    def _get_index_dtype(self, key):

        index = self._get_index(key)
        return index, self._dtypes[index]

    def _expression_getattr(self, expression, attr):

        if attr == 'insert':
            return functools.partial(self._expression_insert, expression)
        else:
            return super()._expression_getattr(expression, attr)

    def _expression_getitem(self, expression, key):

        index, dtype = self._get_index_dtype(key)
        if isinstance(expression, DereferencePointer):
            return DereferencePointer(GetElementPointer(
                expression.address, index, dtype.pointer))
        else:
            return ExtractValue(expression, index, dtype)

#    def _expression_insert(self, expression, value, *indices):
#
#        if isinstance(value, collections.Mapping) and not indices:
#            for indices, value in value.items():
#                if not isinstance(indices, (tuple, list)):
#                    indices = indices,
#                raw_indices = tuple(map(self._get_raw_index, indices))
#                expression = InsertValue(expression, value, raw_indices)
#            return expression
#        else:
#            raw_indices = tuple(map(self._get_raw_index, indices))
#            return InsertValue(expression, value, raw_indices)

    def _assign_helper(self, eb, variable, aggregate_expression):

        if isinstance(self.dtypes, collections.Mapping):
            assert isinstance(aggregate_expression, collections.Mapping)
            for key, value in aggregate_expression.items():
                eb.assign(variable[key], value)
        else:
            if isinstance(aggregate_expression, collections.Mapping):
                for index, value in aggregate_expression.items():
                    eb.assign(variable[index], value)
            else:
                assert isinstance(aggregate_expression, collections.Sequence)
                n = 0
                for index, value in enumerate(aggregate_expression):
                    eb.assign(variable[index], value)
                    n += 1
                assert n == len(self.dtypes)


class IntegerType(FirstClassType):

    def __new__(cls, bits):

        self = super().__new__(cls, 'i{}'.format(bits))
        self.bits = bits
        return self

    def __getnewargs__(self):

        return self.bits,

    def __call__(self, value):

        if isinstance(value, Expression) and value.dtype == self:
            return value
        elif isinstance(value, numbers.Number):
            if isinstance(value, numbers.Integral):
                # TODO: does this always work?
                return LLVMIdentifier(repr(value), self)
            else:
                raise ValueError
        else:
            return self.typecast(self, value)


class SignedIntegerType(IntegerType):

    typecast = MultipleDispatchFunction(2)

    @classmethod
    def smallest_pow2(cls, value):

        assert isinstance(value, numbers.Integral)
        for i in itertools.count(3):
            bits = 2**i
            if -2**(bits-1) <= value < 2**(bits-1):
                return cls(bits)(value)


class UnsignedIntegerType(IntegerType):

    typecast = MultipleDispatchFunction(2)

    @classmethod
    def smallest_pow2(cls, value):

        assert isinstance(value, numbers.Integral)
        if value < 0:
            raise ValueError(
                'cannot represent negative integer as {!r}'.format(cls))
        for i in itertools.count(3):
            bits = 2**i
            if value < 2**bits:
                return cls(bits)(value)


class FloatType(FirstClassType):

    typecast = MultipleDispatchFunction(2)

    def __new__(cls, llvm_id):

        self = super().__new__(cls, llvm_id)
        return self

    def __getnewargs__(self):

        return self._llvm_id,

    def __call__(self, value):

        if isinstance(value, Expression) and value.dtype == self:
            return value
        elif isinstance(value, numbers.Number):
            if isinstance(value, numbers.Real):
                # FIXME: uggly
                value = float(value)
                return LLVMIdentifier(repr(value), self)
            else:
                raise ValueError
        else:
            return self.typecast(self, value)


class Expression:

    def __new__(cls, dtype):

        self = super().__new__(cls)
        self.dtype = dtype
        return self

    @property
    def _llvm_ty_val(self):

        return '{} {}'.format(self.dtype._llvm_id, self._llvm_id)

    def __getattr__(self, attr):

        return self.dtype._expression_getattr(self, attr)

    def __getitem__(self, index):

        return self.dtype._expression_getitem(self, index)

    def __neg__      (self): return neg._operator_call(self)
    def __add__      (l, r): return add._operator_call(l, r)
    def __radd__     (r, l): return add._operator_call(l, r)
    def __sub__      (l, r): return sub._operator_call(l, r)
    def __rsub__     (r, l): return sub._operator_call(l, r)
    def __mul__      (l, r): return mul._operator_call(l, r)
    def __rmul__     (r, l): return mul._operator_call(l, r)
    def __truediv__  (l, r): return truediv._operator_call(l, r)
    def __rtruediv__ (r, l): return truediv._operator_call(l, r)
    def __floordiv__ (l, r): return floordiv._operator_call(l, r)
    def __rfloordiv__(r, l): return floordiv._operator_call(l, r)


class ForceDtype(Expression):

    def __new__(cls, expression, dtype):

        assert isinstance(expression, Expression)

        self = super().__new__(cls, dtype)
        self._expression = expression
        return self

    def _eval(self, bb):

        bb, expression = self._expression._eval(bb)
        expression = LLVMIdentifier(expression._llvm_id, self.dtype)
        return bb, expression


class RHSExpression(Expression):

    def __new__(cls, dtype, rhs_generator, args):

        self = super().__new__(cls, dtype)
        self._rhs_generator = rhs_generator
        self._args = tuple(args)
        return self

    def _eval(self, bb):

        evaluated_args = []
        for arg in self._args:
            bb, arg = arg._eval(bb)
            evaluated_args.append(arg)
        result = bb._reserve_variable(self.dtype)
        bb._append_statement('  {} = {}'.format(
            result._llvm_id, self._rhs_generator(*evaluated_args)))
        return bb, result


class LLVMIdentifier(Expression):

    def __new__(cls, llvm_id, dtype):

        self = super().__new__(cls, dtype)
        self._llvm_id = llvm_id
        return self

    def _eval(self, bb):

        return bb, self


class StringConstant(Expression):

    def __new__(cls, value):

        self = super().__new__(cls, int8_t.pointer)
        self._value = value
        return self

    def _eval(self, bb):

        return bb, bb._module.add_string_constant(self._value)


class PhiNode(LLVMIdentifier):

    def __new__(cls, name, dtype):

        self = super().__new__(cls, name, dtype)
        self._values = []
        return self

    def set_value(self, value, *basic_blocks):

        value = self.dtype(value)
        if not isinstance( value, LLVMIdentifier ):
            raise ValueError( '`value` should be either a variable or a constant, not an expression.' )
        for basic_block in basic_blocks:
            self._values.append((value._llvm_id, basic_block))


class Select(Expression):

    def __new__(cls, condition, value_true, value_false):

        # TODO: allow vectors (also for condition)

        assert value_true.dtype == value_false.dtype
        assert condition._dtype == int1_t

        self = super().__new__(cls, value_true.dtype)
        self._condition = condition
        if not self._condition.dtype == 'i1':
            raise ValueError( "`condition` should have dtype 'i1', got {!r}.".format( self._condition.dtype ) )

        dtype, self._value_true, self._value_false = coerce_default( value_true, value_false )

        Expression.__init__( self, dtype )

    def _eval(self, bb):

        bb, condition = self._condition._eval(bb)
        bb, value_true = self._value_true._eval(bb)
        bb, value_false = self._value_false._eval(bb)

        result = bb._reserve_variable(self.dtype)
        bb._append_statement('  {} = select {}, {}, {}'.format(
            result._llvm_id, condition._llvm_ty_val,
            self._value_true._llvm_ty_val, self._value_false._llvm_ty_val))
        return bb, result


class GetElementPointer(Expression):

    def __new__(cls, pointer, index, dtype):

        if pointer.dtype != dtype and \
                isinstance(pointer.dtype.reference_dtype, StructureType):
            if isinstance(index, numbers.Integral):
                index = int32_t(index)
            else:
                assert isinstance(index, Expression) and index.dtype == int32_t
        else:
            assert isinstance(index, numbers.Integral)  \
                or isinstance(index, Expression) \
                and isinstance(index.dtype, SignedIntegerType)

        if isinstance(pointer, GetElementPointer):
            assert not \
                isinstance(pointer._pointer.dtype.reference_dtype, PointerType)
            # merge nested `GetElementPointer`s
            indices = pointer._indices+(index,)
            pointer = pointer._pointer
        elif dtype == pointer.dtype:
            indices = index,
        else:
            indices = 0, index

        self = super().__new__(cls, dtype)
        self._pointer = pointer
        self._indices = indices
        return self

    def _eval(self, bb):

        bb, pointer = self._pointer._eval(bb)

        statement = []
        for index in self._indices:
            if isinstance(index, numbers.Integral):
                index = bb._module.intptr_t(index)
            bb, index = index._eval(bb)
            statement.append(index._llvm_ty_val)

        result = bb._reserve_variable(self.dtype)
        statement.insert(0, '  {} = getelementptr {}'.format(
            result._llvm_id, pointer._llvm_ty_val))
        bb._append_statement(', '.join(statement))
        return bb, result


class ExtractElement(Expression):

    def __new__(cls, vector, index):

        assert isinstance(vector, Expression) \
            and isinstance(vector.dtype, VectorType)
        # TODO: The documentation is unclear about the signedness of `index`.
        # Assuming signed here.  Find out what llvm uses.
        if isinstance(index, numbers.Integral):
            # TODO: use smallest integer type to represent this number
            index = int32_t(index)
        assert isinstance(index, Expression) \
            and isinstance(index.dtype, SignedIntegerType)

        self = super().__new__(cls, vector.dtype._element_dtype)
        self._vector = vector
        self._index = index
        return self

    def _eval(self, bb):

        bb, vector = self._vector._eval(bb)
        bb, index = self._index._eval(bb)

        result = bb._reserve_variable(self.dtype)
        bb._append_statement('  {} = extractelement {}, {}'.format(
            result._llvm_id, vector._llvm_ty_val, index._llvm_ty_val))
        return bb, result


class InsertElement(Expression):

    def __new__(cls, vector, element, index):

        assert isinstance(vector, Expression) \
            and isinstance(vector.dtype, VectorType)
        element = vector.dtype._element_dtype(element)
        # TODO: The documentation is unclear about the signedness of `index`.
        # Assuming signed here.  Find out what llvm uses.
        if isinstance(index, numbers.Integral):
            # TODO: use smallest integer type to represent this number
            index = int32_t(index)
        assert isinstance(index, Expression) \
            and isinstance(index.dtype, SignedIntegerType)

        self = super().__new__(cls, vector.dtype)
        self._vector = vector
        self._element = element
        self._index = index
        return self

    def _eval(self, bb):

        bb, vector = self._vector._eval(bb)
        bb, element = self._element._eval(bb)
        bb, index = self._index._eval(bb)

        result = bb._reserve_variable(self.dtype)
        bb._append_statement('  {} = insertelement {}, {}, {}'.format(
            result._llvm_id, vector._llvm_ty_val, element._llvm_ty_val,
            index._llvm_ty_val))
        return bb, result


class ExtractValue(Expression):

    def __new__(cls, aggregate_value, index, dtype):

        assert isinstance(index, numbers.Integral)
        indices = index,
        if isinstance(aggregate_value, ExtractValue):
            # merge nested `ExtractValue`s
            indices = aggregate_value._indices+indices
            aggregate_value = aggregate_value._aggregate_value

        self = super().__new__(cls, dtype)
        self._aggregate_value = aggregate_value
        self._indices = indices
        return self

    def _eval(self, bb):

        bb, aggregate_value = self._aggregate_value._eval(bb)

        result = bb._reserve_variable(self.dtype)
        statement = ['  {} = extractvalue {}'.format(
            result._llvm_id, aggregate_value._llvm_id)]
        statement.extend(map(str, self._indices))
        bb._append_statement(', '.join(parts))
        return bb, result


#class InsertValue(Expression):
#
#    def __new__(cls, aggregate_value, element, indices):
#
#        assert len(indices) >= 1
#        element_dtype = aggregate_value.dtype
#        for index in indices:
#            element_dtype = element_dtype._get_element_dtype(index)
#        element = element_dtype(element)
#
#        self = super().__new__(cls, aggregate_value.dtype)
#        self._aggregate_value = aggregate_value
#        self._element = element
#        self._indices = tuple(indices)
#        return self
#
#    def _eval(self, bb):
#
#        bb, aggregate_value = self._aggregate_value._eval(bb)
#        bb, element = self._element._eval(bb)
#
#        result = bb._reserve_variable(self.dtype)
#        indices = ', '.join(map(str, self._indices))
#        bb._append_statement('  {} = insertvalue {}, {}, {}'.format(
#            result._llvm_id, aggregate_value._llvm_ty_val,
#            element._llvm_ty_val, indices))
#        return bb, result


class DereferencePointer(Expression):

    def __new__(cls, address):

        # TODO: align, see
        # http://llvm.org/docs/LangRef.html#load-instruction
        # http://llvm.org/docs/LangRef.html#store-instruction

        self = super().__new__(cls, address.dtype.reference_dtype)
        self.address = address
        return self

    def _eval(self, bb):

        bb, address = self.address._eval(bb)

        result = bb._reserve_variable(self.dtype)
        bb._append_statement('  {} = load {}'.format(
            result._llvm_id, address._llvm_ty_val))
        return bb, result

    def _assign(self, bb, value):

        bb, address = self.address._eval(bb)
        bb, value = value._eval(bb)

        bb._append_statement('  store {}, {}'.format(
            value._llvm_ty_val, address._llvm_ty_val))
        return bb, value


class BasicBlock:

    def __init__(self, label, function_dtype, function_basic_blocks,
            module_ref):

        self._label = label
        self._function_dtype = function_dtype
        self._function_basic_blocks = function_basic_blocks
        self._module_ref = module_ref

        self._phi_nodes = []
        self._statements = []

        self._n_variables = 0
        self._finalised = False

    @property
    def _module(self):

        return self._module_ref()

    def append_basic_block(self):

        return self.append_basic_blocks(1)[0]

    def append_basic_blocks(self, n):

        index = self._function_basic_blocks.index(self)
        labels = tuple(
            LLVMIdentifier(
                '%__L{:04d}'.format(len(self._function_basic_blocks)+i),
                label_t)
            for i in range(n))
        bbs = tuple(
            BasicBlock(label, self._function_dtype,
                self._function_basic_blocks, self._module_ref)
            for label in labels)
        for bb in reversed(bbs):
            self._function_basic_blocks.insert(index+1, bb)
        return bbs

    def _append_statement(self, statement):

        if self._finalised:
            raise ValueError('Cannot append statement to finalised `BasicBlock`.')
        self._statements.append(statement)

    def _generate_ir(self, output):

        if not self._finalised:
            raise ValueError('Incomplete `BasicBlock`.  Add a branch or return statement.')
        print('{}:'.format(self._label._llvm_id[1:]), file=output)
        get_label = lambda bb: bb._tail._label if isinstance(bb, ExtendedBlock) else bb._label
        for phi_node in self._phi_nodes:
            values = ', '.join(
                '[{}, {}]'.format(value, get_label(bb)._llvm_id)
                for value, bb in phi_node._values)
            print('  {} = phi {} {}'.format(phi_node._llvm_id, phi_node.dtype._llvm_id, values), file=output)
        for statement in self._statements:
            print(statement, file=output)
        print(file=output)

    def _reserve_variable(self, dtype):

        name = '{}V{:04d}'.format(self._label._llvm_id, self._n_variables)
        self._n_variables += 1
        return LLVMIdentifier(name, dtype)

    def add_phi_node(self, dtype):

        phi_node = self._reserve_variable(dtype)
        phi_node = PhiNode(phi_node._llvm_id, dtype)
        self._phi_nodes.append(phi_node)
        return phi_node

    def branch1(self, basic_block):

        self._append_statement(
            '  br {}'.format(basic_block._label._llvm_ty_val))
        self._finalised = True

    def branch2(self, condition, bb_true, bb_false):

        self._append_statement('  br {}, {}, {}'.format(
            condition._llvm_ty_val, bb_true._label._llvm_ty_val,
            bb_false._label._llvm_ty_val))
        self._finalised = True

    def ret(self, value=None):

        if self._function_dtype._return_dtype == void_t:
            assert value is None
            self._append_statement('  ret void')
        else:
            assert value is not None
            self._append_statement('  ret {} {}'.format(
                self._function_dtype._return_dtype._llvm_id, value._llvm_id))
        self._finalised = True

    def call(self, function_ptr, *args):

        if isinstance(function_ptr, str):
            function_ptr = self._module._functions[function_ptr]

        assert isinstance(function_ptr.dtype, PointerType) \
            and isinstance(function_ptr.dtype.reference_dtype, FunctionType)
        function_dtype = function_ptr.dtype.reference_dtype

        if function_dtype._variable_arguments:
            function_type = function_dtype._llvm_id+'*'
            assert len(args) >= len(function_dtype._arguments_dtypes)
        else:
            function_type = function_dtype._return_dtype._llvm_id
            assert len(args) == len(function_dtype._arguments_dtypes)

        assert all(isinstance(arg, LLVMIdentifier) for arg in args)
        for arg, dtype in zip(args, function_dtype._arguments_dtypes):
            assert arg.dtype == dtype

        args = ', '.join(
            '{} {}'.format(arg.dtype._llvm_id, arg._llvm_id)
            for arg in args)
        statement = 'call {} {}({})'.format(
            function_type, function_ptr._llvm_id, args)
        if function_dtype._return_dtype == void_t:
            result_var = None
            statement = '  {}'.format(statement)
        else:
            result_var = self._reserve_variable(function_dtype._return_dtype)
            statement = '  {} = {}'.format(result_var._llvm_id, statement)
        self._append_statement(statement)
        return result_var

    def allocate_stack(self, dtype, n_elements=1):

        # TODO: Is `n_elements` allowed to be variable?  Documentation is
        # unclear.
        # http://llvm.org/docs/LangRef.html#alloca-instruction

        result = self._reserve_variable(dtype.pointer)
        statement = [
            '  {} = alloca {}'.format(result._llvm_id, dtype._llvm_id)]
        if n_elements != 1:
            if isinstance(n_elements, numbers.Integral):
                n_elements = int32_t(n_elements)
            assert isinstance(n_elements, LLVMIdentifier) \
                and isinstance(n_elements.dtype, IntegerType)
            # TODO: allow unsigned?
            statement.append(n_elements._llvm_ty_val)
        # TODO: alignment
        self._append_statement(', '.join(statement))
        return result


class ExtendedBlock:

    def __init__(self, entry_basic_block):

        self._head = entry_basic_block
        self._tail = entry_basic_block

    @property
    def _module(self):

        return self._head._module

    @property
    def _finalised(self):

        return self._tail._finalised()

    def append_extended_block(self):

        return self.append_extended_blocks(1)[0]

    def append_extended_blocks(self, n):

        return tuple(map(ExtendedBlock, self._tail.append_basic_blocks(n)))

    def eval(self, expression):

        self._tail, value = expression._eval(self._tail)
        return value

    def assign(self, variable, expression):

        if isinstance(variable.dtype, AggregateType):
            variable.dtype._assign_helper(self, variable, expression)
        else:
            expression = variable.dtype(expression)
            self._tail, expression = variable._assign(self._tail, expression)

    def add_phi_node(self, dtype):

        return self._head.add_phi_node(dtype)

    def branch(self, *extended_blocks):

        # arguments:
        #   last argument should be an ExtendedBlock
        #   all other arguments should be ( condition, extended_block )

        if len(extended_blocks) == 0:
            raise ValueError('at least one extended block should be specified')
        elif len(extended_blocks) == 1:
            self._tail.branch1(extended_blocks[0]._head)
        else:
            condition, eb_true = extended_blocks[0]
            condition = self.eval(condition)
            if len(extended_blocks) > 2:
                eb_false = self.append_extended_block()
            else:
                eb_false = extended_blocks[1]
            self._tail.branch2(condition, eb_true._head, eb_false._head)
            if len(extended_blocks) > 2:
                eb_false.branch(*extended_blocks[1:])

    def ret(self, expression=None):

        if self._head._function_dtype._return_dtype == void_t:
            assert expression is None
            self._tail.ret()
        else:
            assert expression is not None
            expression = self._head._function_dtype._return_dtype(expression)
            expression = self.eval(expression)
            self._tail.ret(expression)

    def call(self, function_ptr, *args):

        if isinstance(function_ptr, str):
            function_ptr = self._module._functions[function_ptr]

        assert isinstance(function_ptr.dtype, PointerType) \
            and isinstance(function_ptr.dtype.reference_dtype, FunctionType)
        function_dtype = function_ptr.dtype.reference_dtype

        if function_dtype._variable_arguments:
            function_type = function_dtype._llvm_id
            assert len(args) >= len(function_dtype._arguments_dtypes)
        else:
            function_type = function_dtype._return_dtype._llvm_id
            assert len(args) == len(function_dtype._arguments_dtypes)

        args = list(args)
        for i in range(len(function_dtype._arguments_dtypes)):
            args[i] = function_dtype._arguments_dtypes[i](args[i])
        args = tuple(map(self.eval, args))

        return self._tail.call(function_ptr, *args)

    def allocate_stack(self, dtype, n_elements=1):

        if isinstance(n_elements, Expression):
            n_elements = self.eval(n_elements)
        return self._tail.allocate_stack(dtype, n_elements=n_elements)


class Module:

    def __init__(self):

        self._header = []
        self._functions = {}
        self._function_definitions = []
        self._string_constants = collections.OrderedDict()

        # TODO: depends on the target platform
        self.size_t = uint64_t
        self.ssize_t = int64_t
        self.intptr_t = int64_t
        self.uintptr_t = uint64_t

        self.declare_function('malloc', [int8_t.pointer, 'noalias'],
            self.size_t, function_attributes=['nounwind'])
        self.declare_function('alligned_alloc', [int8_t.pointer, 'noalias'],
            self.size_t, self.size_t, function_attributes=['nounwind'])
        self.declare_function('free', void_t, [int8_t.pointer, 'nocapture'],
            function_attributes=['nounwind'])
        self.declare_function('printf', int32_t,
            [int8_t.pointer, 'nocapture', 'readonly'], ...)

    @staticmethod
    def _generate_function_type_header(name, return_dtype, arg_dtypes,
            function_attributes, with_arg_expressions):

        if isinstance(return_dtype, (tuple, list)):
            return_dtype, *return_attributes = return_dtype
        else:
            return_attributes = ()
        return_id = ' '.join(list(return_attributes)+[return_dtype._llvm_id])

        if len(arg_dtypes) > 0 and arg_dtypes[-1] == ...:
            arg_dtypes = arg_dtypes[:-1]
            variable_arguments = True
        else:
            variable_arguments = False

        arg_ids, arg_dtypes, _arg_dtypes = [], [], arg_dtypes
        arg_expressions = []
        for i, dtype in enumerate(_arg_dtypes):
            if isinstance(dtype, (tuple, list)):
                dtype, *attributes = dtype
            else:
                attributes = ()
            arg_dtypes.append(dtype)
            if with_arg_expressions:
                arg_expression = \
                    LLVMIdentifier('%__arg_{:04d}'.format(i), dtype)
                attributes += arg_expression._llvm_id,
                arg_expressions.append(arg_expression)
            arg_ids.append(' '.join([dtype._llvm_id]+list(attributes)))
        if variable_arguments:
            arg_ids.append('...')

        function_attributes = ' '+' '.join(function_attributes)
        if function_attributes == ' ':
            function_attributes = ''

        header = '{} @{}({}){}'.format(return_id, name, ', '.join(arg_ids),
            function_attributes)
        dtype = FunctionType(return_dtype, arg_dtypes, variable_arguments)
        function = LLVMIdentifier('@'+name, dtype.pointer)
        if with_arg_expressions:
            return (function, dtype, header)+tuple(arg_expressions)
        else:
            return function, dtype, header

    def declare_function(self, name, return_dtype, *arg_dtypes,
            function_attributes=()):

        function, dtype, header = self._generate_function_type_header(name,
            return_dtype, arg_dtypes, function_attributes, False)
        self._functions[name] = function
        self._header.append('declare '+header)
        return function

    def define_function(self, name, return_dtype, *arg_dtypes,
            function_attributes=()):

        function, dtype, header, *args = self._generate_function_type_header(
            name, return_dtype, arg_dtypes, function_attributes, True)
        self._functions[name] = function
        function_basic_blocks = []
        entry = BasicBlock(LLVMIdentifier('%__L0000', label_t), dtype,
            function_basic_blocks, weakref.ref(self))
        function_basic_blocks.append(entry)
        self._function_definitions.append((header, entry))
        return (function, ExtendedBlock(entry))+tuple(args)

    def add_string_constant(self, data):

        assert isinstance(data, bytes)
        try:
            return self._string_constants[data]
        except KeyError:
            pass

        len_data = len(data)+1
        var = '@__string_constant_{:04d}'.format(len(self._string_constants))
        enc = ''.join(map('\\{:02X}'.format, data))
        self._header.append('{} = private unnamed_addr constant [{} x i8] c"{}\\00", align 1'.format(var, len_data, enc))
        var = LLVMIdentifier(
            'getelementptr inbounds ([{} x i8]* {}, i32 0, i32 0)'.format(
                len(data)+1, var),
            int8_t.pointer)
        self._string_constants[data] = var
        return  var

    def _generate_ir(self, output):

        for line in self._header:
            print(line, file=output)
        print()
        for header, bb_entry in self._function_definitions:
            print('define {} {{'.format(header), file=output)
            for basic_block in bb_entry._function_basic_blocks:
                basic_block._generate_ir(output=output)
            print('}', file=output)
            print(file=output)

    def sizeof(self, dtype):

        # TODO: size: portable solution is given at
        # http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt
        # also see http://llvm.org/docs/LangRef.html#data-layout
        if isinstance(dtype, PointerType):
            return 8
        elif isinstance(dtype, IntegerType):
            return (dtype.bits+7)//8
        elif isinstance(dtype, FloatType):
            if dtype._llvm_id == 'half':
                return 2
            elif dtype._llvm_id == 'float':
                return 4
            elif dtype._llvm_id == 'double':
                return 8
            else:
                raise ValueError
        else:
            raise ValueError


neg = MultipleDispatchFunction(1)
add = MultipleDispatchFunction(2)
sub = MultipleDispatchFunction(2)
mul = MultipleDispatchFunction(2)
truediv = MultipleDispatchFunction(2)
floordiv = MultipleDispatchFunction(2)
lt = MultipleDispatchFunction(2)
le = MultipleDispatchFunction(2)
gt = MultipleDispatchFunction(2)
ge = MultipleDispatchFunction(2)
eq = MultipleDispatchFunction(2)
ne = MultipleDispatchFunction(2)




void_t = VoidType()
label_t = LabelType()

int1_t = SignedIntegerType(1)

int8_t = SignedIntegerType(8)
uint8_t = UnsignedIntegerType(8)
int16_t = SignedIntegerType(16)
uint16_t = UnsignedIntegerType(16)
int32_t = SignedIntegerType(32)
uint32_t = UnsignedIntegerType(32)
int64_t = SignedIntegerType(64)
uint64_t = UnsignedIntegerType(64)
float16_t = FloatType('half')
float32_t = FloatType('float')
float64_t = FloatType('double')


def _gen_bin_op(dtype_class, py_dtype, llvm_op, return_dtype=None):
    def custom(l, r):
        _return_dtype = return_dtype
        if isinstance(l, Expression) and isinstance(l.dtype, dtype_class):
            if isinstance(r, Expression) and l.dtype == r.dtype:
                pass
            elif isinstance(r, py_dtype):
                r = l.dtype(r)
            else:
                return NotImplemented
        elif isinstance(r, Expression) and isinstance(r.dtype, dtype_class) \
                and isinstance(l, py_dtype):
            l = r.dtype(l)
        elif isinstance(l, Expression) and isinstance(r, Expression) \
                and isinstance(l.dtype, VectorType) \
                and isinstance(l.dtype._element_dtype, dtype_class) \
                and r.dtype == l.dtype:
            if _return_dtype is not None:
                _return_dtype = VectorType(_return_dtype)
        else:
            return NotImplemented
        return RHSExpression(
            l.dtype if _return_dtype is None else _return_dtype,
            lambda _l, _r: '{} {}, {}'.format(
                llvm_op, _l._llvm_ty_val, _r._llvm_id),
            (l, r))
    return custom

for _dtype_class, _flag in \
        ((SignedIntegerType, 's'), (UnsignedIntegerType, 'u')):
    for _op, _llvm_op in ((add, 'add'), (sub, 'sub'), (mul, 'mul'),
            (floordiv, _flag+'div')):
        _op.register(_gen_bin_op(_dtype_class, numbers.Integral, _llvm_op))
    for _op, _llvm_op in ((lt, _flag+'lt'), (le, _flag+'le'), (gt, _flag+'gt'),
            (ge, _flag+'ge'), (eq, 'eq'), (ne, 'ne')):
        _op.register(_gen_bin_op(
            _dtype_class, numbers.Integral, 'icmp '+_llvm_op, int1_t))

for _op, _llvm_op in ((add, 'fadd'), (sub, 'fsub'), (mul, 'fmul'),
        (truediv, 'fdiv')):
    _op.register(_gen_bin_op(FloatType, numbers.Real, _llvm_op))
# TODO: support the unordered comparisons?
#       see http://llvm.org/docs/LangRef.html#fcmp-instruction
for _op, _llvm_op in ((lt, 'olt'), (le, 'ole'), (gt, 'ogt'), (ge, 'oge'),
        (eq, 'oeq'), (ne, 'one')):
    _op.register(_gen_bin_op(
        FloatType, numbers.Real, 'fcmp '+_llvm_op, int1_t))

@neg.register
def _neg_custom(value):
    if isinstance(value, Expression) and \
            (isinstance(value.dtype, (IntegerType, FloatType)) \
            or isinstance(value.dtype, VectorType) \
            and isinstance(value.dtype._element_dtype,
                (IntegerType, FloatType))):
        return 0-value
    else:
        return NotImplemented

@SignedIntegerType.typecast.register
def _typecast(new_dtype, value):
    if isinstance(value.dtype, UnsignedIntegerType):
        value = ForceDtype(value, SignedIntegerType(value.dtype.bits))
    old_dtype = value.dtype
    if old_dtype == new_dtype:
        return value
    elif isinstance(old_dtype, SignedIntegerType):
        if new_dtype.bits > old_dtype.bits:
            op = 'sext'
        elif new_dtype.bits < old_dtype.bits:
            op = 'trunc'
        else:
            raise ValueError
    elif isinstance(old_dtype, FloatType):
        op = 'fptosi'
    else:
        return NotImplemented
    return RHSExpression(
        new_dtype,
        lambda _value:
            '{} {} to {}'.format(op, _value._llvm_ty_val, new_dtype._llvm_id),
        (value,))

@UnsignedIntegerType.typecast.register
def _typecast(new_dtype, value):
    if isinstance(value.dtype, SignedIntegerType):
        value = ForceDtype(value, UnsignedIntegerType(value.dtype.bits))
    old_dtype = value.dtype
    if old_dtype == new_dtype:
        return value
    elif isinstance(old_dtype, UnsignedIntegerType):
        if new_dtype.bits > old_dtype.bits:
            op = 'zext'
        elif new_dtype.bits < old_dtype.bits:
            op = 'trunc'
        else:
            raise ValueError
    elif isinstance(old_dtype, FloatType):
        op = 'fptoui'
    else:
        return NotImplemented
    return RHSExpression(
        new_dtype,
        lambda _value:
            '{} {} to {}'.format(op, _value._llvm_ty_val, new_dtype._llvm_id),
        (value,))

@FloatType.typecast.register
def _typecast(new_dtype, value):
    old_dtype = value.dtype
    if old_dtype == new_dtype:
        return value
    elif isinstance(old_dtype, FloatType):
        fp_order = ('half', 'float', 'double')
        if old_dtype._llvm_id in fp_order and new_dtype._llvm_id in fp_order:
            old_index = fp_order.index(old_dtype._llvm_id)
            new_index = fp_order.index(new_dtype._llvm_id)
            if old_index < new_index:
                op = 'fpext'
            else:
                op = 'fptrunc'
        else:
            return NotImplemented
    elif isinstance(old_dtype, SignedIntegerType):
        op = 'sitofp'
    elif isinstance(old_dtype, UnsignedIntegerType):
        op = 'uitofp'
    else:
        return NotImplemented
    return RHSExpression(
        new_dtype,
        lambda _value:
            '{} {} to {}'.format(op, _value._llvm_ty_val, new_dtype._llvm_id),
        (value,))

del _dtype_class, _flag, _op, _llvm_op, _gen_bin_op, _neg_custom, _typecast

# vim: ts=4:sts=4:sw=4:et
