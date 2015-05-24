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

        if self.test_dtype(expression):
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

    def test_dtype(self, value):

        return isinstance(value, Expression) and value.dtype == self

    @classmethod
    def test_dtype_class(cls, value):

        return isinstance(value, Expression) and isinstance(value.dtype, cls)


class VoidType(TypeBase):

    def __new__(cls):

        return super().__new__(cls, 'void')

    def __getnewargs__(self):

        return ()


class TypeAttributesTuple:

    def __new__(cls, value):

        if isinstance(value, cls):
            return value

        self = super().__new__(cls)
        if isinstance(value, TypeBase):
            self.dtype = value
            self.attributes = frozenset()
        else:
            self.dtype, *self.attributes = value
            self.attributes = frozenset(self.attributes)
            assert isinstance(self.dtype, TypeBase)
            assert all(isinstance(a, str) for a in self.attributes)
        return self

    def __eq__(self, other):

        return self.__class__ == other.__class__ \
            and self.dtype == other.dtype \
            and self.attributes == other.attributes

    def __hash__(self):

        return hash((self.dtype, self.attributes))

    def filter_attributes(self, whitelist):

        return TypeAttributesTuple(
            (self.dtype,)+tuple(self.attributes&whitelist))


class FunctionType(TypeBase):

    # TODO: add `returns_twice` to `_function_attributes`?
    _allowed_function_attributes = frozenset(('noreturn', 'nounwind',
        'readnone', 'readonly'))
    _allowed_parameter_attributes = frozenset(('readnone', 'readonly',
        'noalias', 'nocapture', 'nonnull'))
    _allowed_return_attributes = frozenset(('noalias', 'nonnull'))

    def __new__(cls, return_value, parameters, variable_arguments,
            function_attributes):

        return_value = TypeAttributesTuple(return_value).filter_attributes(
            cls._allowed_return_attributes)
        parameters = tuple(
            TypeAttributesTuple(parameter).filter_attributes(
                cls._allowed_parameter_attributes)
            for parameter in parameters)
        variable_arguments = bool(variable_arguments)
        function_attributes = \
            frozenset(function_attributes)&cls._allowed_function_attributes

        param_ids = [p.dtype._llvm_id for p in parameters]
        if variable_arguments:
            param_ids.append('...')
        llvm_id = '{} ({})'.format(
            return_value.dtype._llvm_id, ', '.join(param_ids))

        self = super().__new__(cls, llvm_id)
        self._return_value = return_value
        self._parameters = parameters
        self._variable_arguments = variable_arguments
        self._function_attributes = function_attributes
        return self

    def __getnewargs__(self):

        return self._return_value, self._parameters, self._variable_arguments,\
            self._function_attributes


class FirstClassType(TypeBase):

    def bitcast(self, value):

        if not isinstance(value, Expression):
            raise ValueError('Cannot bitcast {!r} to {!r}'.format(value, self))
        elif value.dtype == self:
            return value
        elif isinstance(value.dtype, FirstClassType):
            return FormatExpression(self, 'bitcast {0:dtype} {0} to {dtype}',
                (value,))
        else:
            raise ValueError(
                'Can only bitcast first class types, not {!r}'.format(
                    value.dtype))


class PrimitiveType(FirstClassType):

    def test_dtype_or_vector(self, value):

        return isinstance(value, Expression) and ( \
            value.dtype == self \
            or isinstance(value.dtype, VectorType) \
            and value.dtype._element_dtype == self)

    @classmethod
    def test_dtype_class_or_vector(cls, value):

        return isinstance(value, Expression) and ( \
            isinstance(value.dtype, cls)
            or isinstance(value.dtype, VectorType) \
            and isinstance(value.dtype._element_dtype, cls))

    def vector(self, length):

        return VectorType(self, length)


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

    def _expression_call(self, expression, args, kwargs):

        if isinstance(self.reference_dtype, FunctionType):
            assert not kwargs
            if self.reference_dtype._function_attributes\
                    &frozenset(('readnone', 'readonly')):
                return ReadOnlyFunctionCall(expression, args)
            else:
                return FunctionCall(expression, args)
        else:
            return super()._expresion_call(expression, args, kwargs)


class VectorType(FirstClassType):

    def __new__(cls, dtype, length):

        assert isinstance(dtype, PrimitiveType)
        self = super().__new__(cls, '<{} x {}>'.format(length, dtype._llvm_id))
        self._element_dtype = dtype
        self._length = length
        return self

    def __getnewargs__(self):

        return self._element_dtype, self._length

    def __call__(self, elements):

        # TODO: typecasting
        if self.test_dtype(elements):
            return elements
        else:
            return self.create(elements)

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


class IntegerType(PrimitiveType):

    def __new__(cls, bits):

        self = super().__new__(cls, 'i{}'.format(bits))
        self.bits = bits
        return self

    def __getnewargs__(self):

        return self.bits,

    def __call__(self, value):

        if self.test_dtype(value):
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


class FloatType(PrimitiveType):

    typecast = MultipleDispatchFunction(2)

    def __new__(cls, llvm_id):

        self = super().__new__(cls, llvm_id)
        return self

    def __getnewargs__(self):

        return self._llvm_id,

    def __call__(self, value):

        if self.test_dtype(value):
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

    def __new__(cls, dtype, children):

        self = super().__new__(cls)
        self.dtype = dtype
        self._children = tuple(children)
        return self

    def _eval_tree(self, bb, cache):

        try:
            return bb, cache[self]
        except KeyError:
            pass

        children = []
        for child in self._children:
            bb, child = child._eval_tree(bb, cache)
            children.append(child)
        bb, value = self._eval(bb, *children)
        cache[self] = value
        return bb, value

    @property
    def _llvm_ty_val(self):

        return '{} {}'.format(self.dtype._llvm_id, self._llvm_id)

    def __getattr__(self, attr):

        return self.dtype._expression_getattr(self, attr)

    def __getitem__(self, index):

        return self.dtype._expression_getitem(self, index)

    def __call__(*args, **kwargs):

        self, *args = args
        return self.dtype._expression_call(self, args, kwargs)

    def __neg__(self):

        value = neg._operator_call(self)
        if value == NotImplemented:
            raise TypeError
        else:
            return value

    def __abs__(self):

        value = abs_._operator_call(self)
        if value == NotImplemented:
            raise TypeError
        else:
            return value

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
    def __mod__      (l, r): return mod._operator_call(l, r)
    def __rmod__     (r, l): return mod._operator_call(l, r)


class ForceDtype(Expression):

    def __new__(cls, expression, dtype):

        assert isinstance(expression, Expression)

        self = super().__new__(cls, dtype, (expression,))
        return self

    def _eval(self, bb, expression):

        return bb, LLVMIdentifier(expression._llvm_id, self.dtype)


class FormatExpressionArg:

    def __new__(cls, arg):

        self = super().__new__(cls)
        self.arg = arg
        return self

    def __str__(self):

        return self.arg._llvm_id

    def __format__(self, fmt):

        if fmt == '':
            return self.arg._llvm_id
        elif fmt == 'dtype':
            return self.arg.dtype._llvm_id
        else:
            raise ValueError('Invalid format specifier')


class FormatExpression(Expression):

    def __new__(cls, dtype, expression_fmt, children):

        self = super().__new__(cls, dtype, children)
        self._expression_fmt = expression_fmt
        return self

    def _eval(self, bb, *args):

        expression_args = map(FormatExpressionArg, args)
        kwargs = {'dtype': self.dtype._llvm_id}
        expression = self._expression_fmt.format(*expression_args, **kwargs)
        return bb, bb._append_rhs_statement(self.dtype, expression)


class LLVMIdentifier(Expression):

    def __new__(cls, llvm_id, dtype):

        self = super().__new__(cls, dtype, ())
        self._llvm_id = llvm_id
        return self

    def _eval(self, bb):

        return bb, self


class StringConstant(Expression):

    def __new__(cls, value):

        self = super().__new__(cls, int8_t.pointer, ())
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


def Select(condition, value_true, value_false):

    value_true, value_false = _convert_python_numbers(value_true, value_false)
    assert int1_t.test_dtype_or_vector(condition)
    assert isinstance(value_true, Expression)
    assert isinstance(value_false, Expression)
    assert value_true.dtype == value_false.dtype
    if isinstance(condition.dtype, VectorType):
        assert isinstance(value_true.dtype, VectorType)
        assert value_true.dtype._length == condition.dtype._length

    return FormatExpression(
        value_true.dtype,
        'select {0:dtype} {0}, {1:dtype} {1}, {2:dtype} {2}',
        (condition, value_true, value_false))


class GetElementPointer(Expression):

    def __new__(cls, pointer, index, dtype):

        if pointer.dtype != dtype and \
                isinstance(pointer.dtype.reference_dtype, StructureType):
            if isinstance(index, numbers.Integral):
                index = int32_t(index)
            else:
                assert isinstance(index, Expression) and index.dtype == int32_t
        else:
            if isinstance(index, numbers.Integral):
                index = SignedIntegerType.smallest_pow2(index)
            else:
                assert SignedIntegerType.test_dtype_class(index)

        if isinstance(pointer, GetElementPointer):
            assert not \
                isinstance(pointer._pointer.dtype.reference_dtype, PointerType)
            # merge nested `GetElementPointer`s
            indices = pointer._indices+(index,)
            pointer = pointer._pointer
        elif dtype == pointer.dtype:
            indices = index,
        else:
            indices = int8_t(0), index

        self = super().__new__(cls, dtype, (pointer,)+indices)
        self._pointer = pointer
        self._indices = indices
        return self

    def _eval(self, bb, pointer, *indices):

        result = bb._reserve_variable(self.dtype)
        statement = ['  {} = getelementptr {}'.format(
            result._llvm_id, pointer._llvm_ty_val)]
        statement.extend(index._llvm_ty_val for index in indices)
        bb._append_statement(', '.join(statement))
        return bb, result


def ExtractElement(vector, index):

    assert VectorType.test_dtype_class(vector)
    # TODO: The documentation is unclear about the signedness of `index`.
    # Assuming signed here.  Find out what llvm uses.
    if isinstance(index, numbers.Integral):
        index = SignedIntegerType.smallest_pow2(index)
    else:
        assert SignedIntegerType.test_dtype_class(index)

    return FormatExpression(
        vector.dtype._element_dtype,
        'extractelement {0:dtype} {0}, {1:dtype} {1}',
        (vector, index))


def InsertElement(vector, element, index):

    assert VectorType.test_dtype_class(vector)
    element = vector.dtype._element_dtype(element)
    # TODO: The documentation is unclear about the signedness of `index`.
    # Assuming signed here.  Find out what llvm uses.
    if isinstance(index, numbers.Integral):
        index = SignedIntegerType.smallest_pow2(index)
    else:
        assert SignedIntegerType.test_dtype_class(index)

    return FormatExpression(
        vector.dtype,
        'insertelement {0:dtype} {0}, {1:dtype} {1}, {2:dtype} {2}',
        (vector, element, index))


class ExtractValue(Expression):

    def __new__(cls, aggregate_value, index, dtype):

        assert isinstance(index, numbers.Integral)
        indices = index,
        if isinstance(aggregate_value, ExtractValue):
            # merge nested `ExtractValue`s
            indices = aggregate_value._indices+indices
            aggregate_value = aggregate_value._aggregate_value

        self = super().__new__(cls, dtype, (aggregate_value,))
        self._indices = indices
        return self

    def _eval(self, bb, aggregate_value):

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

        self = super().__new__(cls, address.dtype.reference_dtype, (address,))
        self.address = address
        return self

    def _eval(self, bb, address):

        return bb, bb._append_rhs_statement(self.dtype, 'load {} {}'.format(
            address.dtype._llvm_id, address._llvm_id))


class FunctionCall:

    def __new__(cls, function_pointer, args):

        func_dtype = function_pointer.dtype.reference_dtype
        dtypes = itertools.chain(
            (param.dtype for param in func_dtype._parameters),
            itertools.cycle([lambda e: e]))
        args = tuple(dtype(arg) for arg, dtype in zip(args, dtypes))

        self = super().__new__(cls)
        self._children = (function_pointer,)+args
        return self


class ReadOnlyFunctionCall(Expression):

    def __new__(cls, function_pointer, args):

        func_dtype = function_pointer.dtype.reference_dtype
        children = [function_pointer]
        dtypes = itertools.chain(
            (param.dtype for param in func_dtype._parameters),
            itertools.cycle([lambda e: e]))
        children.extend(dtype(arg) for arg, dtype in zip(args, dtypes))

        assert func_dtype._return_value.dtype != void_t

        self = super().__new__(cls, func_dtype._return_value.dtype, children)
        self._function_dtype = func_dtype
        return self

    def _eval(self, bb, *children):

        return bb, bb.call(*children)


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

    def _append_rhs_statement(self, dtype, rhs):

        var = self._reserve_variable(dtype)
        self._append_statement('  {} = {}'.format(var._llvm_id, rhs))
        return var

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

        if self._function_dtype._return_value.dtype == void_t:
            assert value is None
            self._append_statement('  ret void')
        else:
            assert value is not None
            self._append_statement('  ret {} {}'.format(
                self._function_dtype._return_value.dtype._llvm_id,
                value._llvm_id))
        self._finalised = True

    def call(*args):

        self, function_pointer, *args = args

        if isinstance(function_pointer, str):
            function_pointer = self._module._functions[function_pointer]

        assert isinstance(function_pointer, LLVMIdentifier)
        assert all(isinstance(arg, LLVMIdentifier) for arg in args)

        function_dtype = function_pointer.dtype.reference_dtype
        if function_dtype._variable_arguments:
            function_type = function_dtype._llvm_id+'*'
            assert len(args) >= len(function_dtype._parameters)
        else:
            function_type = function_dtype._return_value.dtype._llvm_id
            assert len(args) == len(function_dtype._parameters)

        # TODO: check argument dtypes

        return_dtype = function_dtype._return_value.dtype
        if return_dtype == void_t:
            result_var = None
            prefix = '  '
        else:
            result_var = self._reserve_variable(return_dtype)
            prefix = '  {} = '.format(result_var._llvm_id)

        self._append_statement('{}call {} {}({})'.format(
            prefix, function_type, function_pointer._llvm_id,
            ', '.join(arg._llvm_ty_val for arg in args)))
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

    def store(self, address, value):

        self._append_statement('  store {}, {}'.format(
            value._llvm_ty_val, address._llvm_ty_val))


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

        if isinstance(expression, FunctionCall):
            # nested FunctionCalls are not allowed
            return self._tail.call(*self.eval(expression._children))
        elif isinstance(expression, Expression):
            return self.eval((expression,))[0]
        elif isinstance(expression, (tuple, list)):
            cache = {}
            values = []
            for expression in expression:
                if not isinstance(expression, Expression):
                    raise ValueError(
                        'not an expression: {!r}'.format(expression))
                self._tail, value = expression._eval_tree(self._tail, cache)
                values.append(value)
            return values
        else:
            raise ValueError('`eval` takes an `Expression` or a list of '\
                '`Expression`s, not {!r}.'.format(expression))

    def assign(self, variable, expression):

        if isinstance(variable.dtype, AggregateType):
            variable.dtype._assign_helper(self, variable, expression)
        else:
            expression = variable.dtype(expression)
            address, value = self.eval((variable.address, expression))
            return self._tail.store(address, value)

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

        if self._head._function_dtype._return_value.dtype == void_t:
            assert expression is None
            self._tail.ret()
        else:
            assert expression is not None
            expression = self._head._function_dtype._return_value.dtype(
                expression)
            expression = self.eval(expression)
            self._tail.ret(expression)

    def call(*args, **kwargs):

        self, function, *args = args
        if isinstance(function, str):
            function = self._module._functions[function]
        return self.eval(function(*args, **kwargs))

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
        self._link_ir = []

        self.declare_function('llvm.fabs.f32', float32_t, float32_t)
        self.declare_function('llvm.fabs.f64', float64_t, float64_t)
        self.declare_function('llvm.copysign.f32', float32_t, float32_t,
            float32_t)
        self.declare_function('llvm.copysign.f64', float64_t, float64_t,
            float64_t)
        self.declare_function('llvm.floor.f32', float32_t, float32_t)
        self.declare_function('llvm.floor.f64', float64_t, float64_t)
        self.declare_function('llvm.ceil.f32', float32_t, float32_t)
        self.declare_function('llvm.ceil.f64', float64_t, float64_t)

    @staticmethod
    def _generate_function_type_header(name, return_value, parameters,
            function_attributes, with_arg_expressions):

        if parameters and parameters[-1] == ...:
            variable_arguments = True
            parameters = parameters[:-1]
        else:
            variable_arguments = False

        return_value = TypeAttributesTuple(return_value)
        parameters = tuple(map(TypeAttributesTuple, parameters))
        function_attributes = frozenset(function_attributes)

        dtype = FunctionType(return_value, parameters, variable_arguments,
            function_attributes)

        return_id = ' '.join(itertools.chain(
            sorted(return_value.attributes), (return_value.dtype._llvm_id,)))

        arg_ids = []
        arg_expressions = []
        for i, parameter in enumerate(parameters):
            parts = [parameter.dtype._llvm_id]
            parts.extend(sorted(parameter.attributes))
            if with_arg_expressions:
                arg_expression = \
                    LLVMIdentifier('%__arg_{:04d}'.format(i), parameter.dtype)
                parts.append(arg_expression._llvm_id)
                arg_expressions.append(arg_expression)
            arg_ids.append(' '.join(parts))
        if variable_arguments:
            arg_ids.append('...')

        function_attributes = ' '+' '.join(sorted(function_attributes))
        if function_attributes == ' ':
            function_attributes = ''

        header = '{} @{}({}){}'.format(return_id, name, ', '.join(arg_ids),
            function_attributes)
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
        print(file=output)
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
rtzdiv = MultipleDispatchFunction(2)
mod = MultipleDispatchFunction(2)
rtzmod = MultipleDispatchFunction(2)
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


def _convert_python_numbers(*args):

    dtypes = set(arg.dtype for arg in args if isinstance(arg, Expression))
    if len(dtypes) != 1:
        return args
    dtype = next(iter(dtypes))
    try:
        return tuple(
            dtype(arg) if isinstance(arg, numbers.Number) else arg
            for arg in args)
    except TypeError:
        pass


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
        return FormatExpression(
            l.dtype if _return_dtype is None else _return_dtype,
            llvm_op+' {0:dtype} {0}, {1}', (l, r))
    return custom

for _dtype_class, _flag in \
        ((SignedIntegerType, 's'), (UnsignedIntegerType, 'u')):
    for _op, _llvm_op in ((add, 'add'), (sub, 'sub'), (mul, 'mul')):
        _op.register(_gen_bin_op(_dtype_class, numbers.Integral, _llvm_op))
    for _op, _llvm_op in ((lt, _flag+'lt'), (le, _flag+'le'), (gt, _flag+'gt'),
            (ge, _flag+'ge'), (eq, 'eq'), (ne, 'ne')):
        _op.register(_gen_bin_op(
            _dtype_class, numbers.Integral, 'icmp '+_llvm_op, int1_t))

rtzdiv.register(_gen_bin_op(SignedIntegerType, numbers.Integral, 'sdiv'))
rtzdiv.register(_gen_bin_op(UnsignedIntegerType, numbers.Integral, 'udiv'))
floordiv.register(_gen_bin_op(UnsignedIntegerType, numbers.Integral, 'udiv'))

rtzmod.register(_gen_bin_op(SignedIntegerType, numbers.Integral, 'srem'))
rtzmod.register(_gen_bin_op(UnsignedIntegerType, numbers.Integral, 'urem'))
mod.register(_gen_bin_op(UnsignedIntegerType, numbers.Integral, 'urem'))

for _op, _llvm_op in ((add, 'fadd'), (sub, 'fsub'), (mul, 'fmul'),
        (truediv, 'fdiv'), (mod, 'frem')):
    _op.register(_gen_bin_op(FloatType, numbers.Real, _llvm_op))
# TODO: support the unordered comparisons?
#       see http://llvm.org/docs/LangRef.html#fcmp-instruction
for _op, _llvm_op in ((lt, 'olt'), (le, 'ole'), (gt, 'ogt'), (ge, 'oge'),
        (eq, 'oeq'), (ne, 'one')):
    _op.register(_gen_bin_op(
        FloatType, numbers.Real, 'fcmp '+_llvm_op, int1_t))

@floordiv.register
def _operator(l, r):
    if isinstance(l, Expression) and isinstance(l.dtype, SignedIntegerType):
        if isinstance(r, Expression) and l.dtype == r.dtype:
            pass
        elif isinstance(r, py_dtype):
            r = l.dtype(r)
        else:
            return NotImplemented
    elif isinstance(r, Expression) and isinstance(r.dtype, SignedIntegerType) \
            and isinstance(l, py_dtype):
        l = r.dtype(l)
    elif isinstance(l, Expression) and isinstance(r, Expression) \
            and isinstance(l.dtype, VectorType) \
            and isinstance(l.dtype._element_dtype, SignedIntegerType) \
            and r.dtype == l.dtype:
        if return_dtype is not None:
            return_dtype = VectorType(return_dtype)
    else:
        return NotImplemented
    return Select(ge(l*r, 0), rtzdiv(l, r),
        Select(ge(l, 0), rtzdiv(l-1, r)-1, rtzdiv(l+1, r)-1))

@mod.register
def _operator(l, r):
    if isinstance(l, Expression) and isinstance(l.dtype, SignedIntegerType):
        if isinstance(r, Expression) and l.dtype == r.dtype:
            pass
        elif isinstance(r, py_dtype):
            r = l.dtype(r)
        else:
            return NotImplemented
    elif isinstance(r, Expression) and isinstance(r.dtype, SignedIntegerType) \
            and isinstance(l, py_dtype):
        l = r.dtype(l)
    elif isinstance(l, Expression) and isinstance(r, Expression) \
            and isinstance(l.dtype, VectorType) \
            and isinstance(l.dtype._element_dtype, SignedIntegerType) \
            and r.dtype == l.dtype:
        if return_dtype is not None:
            return_dtype = VectorType(return_dtype)
    else:
        return NotImplemented
    return Select(ge(r, 0),
        Select(ge(l, 0), rtzmod(l, r), rtzmod(r-rtzmod(-l, r), r)),
        Select(ge(l, 0), -rtzmod(-r-rtzmod(l, -r), -r), -rtzmod(-l, -r)))

@neg.register
def _neg_custom(value):
    if isinstance(value, Expression) and \
            (isinstance(value.dtype, (SignedIntegerType, FloatType)) \
            or isinstance(value.dtype, VectorType) \
            and isinstance(value.dtype._element_dtype,
                (SignedIntegerType, FloatType))):
        return 0-value
    else:
        return NotImplemented

@rtzmod.register
def _operator(l, r):
    if isinstance(l, Expression) and isinstance(l.dtype, FloatType):
        if isinstance(r, Expression) and l.dtype == r.dtype:
            pass
        elif isinstance(r, py_dtype):
            r = l.dtype(r)
        else:
            return NotImplemented
    elif isinstance(r, Expression) and isinstance(r.dtype, FloatType) \
            and isinstance(l, py_dtype):
        l = r.dtype(l)
    elif isinstance(l, Expression) and isinstance(r, Expression) \
            and isinstance(l.dtype, VectorType) \
            and isinstance(l.dtype._element_dtype, FloatType) \
            and r.dtype == l.dtype:
        pass
    else:
        return NotImplemented
    return copysign(abs(l)%abs(r), l)

@floordiv.register
def _operator(l, r):
    value = truediv._operator_call(l, r)
    if value is NotImplemented:
        return NotImplemented
    elif isinstance(value.dtype, FloatType):
        el_dtype = value.dtype
    elif isinstance(value.dtype, VectorType) and \
            isinstance(value.dtype._element_dtype, FloatType):
        el_dtype = value._element_dtype.dtype
    else:
        return NotImplemented
    return floor(value)

@rtzdiv.register
def _operator(l, r):
    value = truediv._operator_call(l, r)
    if value is NotImplemented:
        return NotImplemented
    elif isinstance(value.dtype, FloatType):
        el_dtype = value.dtype
    elif isinstance(value.dtype, VectorType) and \
            isinstance(value.dtype._element_dtype, FloatType):
        el_dtype = value._element_dtype.dtype
    else:
        return NotImplemented
    return Select(ge(value, 0), floor(value), ceil(value))

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
    return FormatExpression(
        new_dtype, op+' {0:dtype} {0} to {dtype}', (value,))

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
    return FormatExpression(
        new_dtype, op+' {0:dtype} {0} to {dtype}', (value,))

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
    return FormatExpression(
        new_dtype, op+' {0:dtype} {0} to {dtype}', (value,))

del _dtype_class, _flag, _op, _llvm_op, _gen_bin_op, _neg_custom, _typecast
del _operator


class _ReadOnlyIntrinsicFunctionCall(ReadOnlyFunctionCall):

    def _eval(self, bb, *children):

        if children[0]._llvm_id[1:] not in bb._module._functions:
            dtype = children[0].dtype.reference_dtype
            assert not dtype._variable_arguments
            bb._module.declare_function(
                children[0]._llvm_id[1:],
                dtype._return_value,
                *dtype._parameters,
                function_attributes=dtype._function_attributes)
        return super()._eval(bb, *children)


def _gen_float_intrinsic(name, n_args):

    func = MultipleDispatchFunction(n_args)

    @func.register
    def _llvm_intrinsic(*args):

        # `len(args) == n_args` is asserted by `MultipleDispatchFunction`

        # list all floating point types
        float_dtypes = set(
            arg.dtype
            for arg in args
            if FloatType.test_dtype_class_or_vector(arg))
        # extract the *single* float dtype
        if len(float_dtypes) != 1:
            return NotImplemented
        float_dtype = next(iter(float_dtypes))
        # convert all args to float
        try:
            args = tuple(map(float_dtype, args))
        except NotImplementedError:
            return NotImplemented
        # determine float name
        if isinstance(float_dtype, VectorType):
            scalar_float_dtype = float_dtype._element_dtype
            vector_name = 'v{}'.format(float_dtype._length)
        else:
            scalar_float_dtype = float_dtype
            vector_name = ''
        _float_names = {
            float32_t: 'f32',
            float64_t: 'f64',
        }
        float_name = _float_names[scalar_float_dtype]
        # call intrinsic
        func_dtype = FunctionType(
            float_dtype, (float_dtype,)*n_args, False, ())
        scalar_func_dtype = FunctionType(
            scalar_float_dtype, (scalar_float_dtype,)*n_args, False, ())
        func = LLVMIdentifier(
            '@llvm.{}.{}{}'.format(name, vector_name, float_name),
            func_dtype.pointer)
        return _ReadOnlyIntrinsicFunctionCall(func, args)

    return func


sqrt = _gen_float_intrinsic('sqrt', 1)
sin = _gen_float_intrinsic('sin', 1)
cos = _gen_float_intrinsic('cos', 1)
exp = _gen_float_intrinsic('exp', 1)
exp2 = _gen_float_intrinsic('exp2', 1)
log = _gen_float_intrinsic('log', 1)
log10 = _gen_float_intrinsic('log10', 1)
log2 = _gen_float_intrinsic('log2', 1)
abs_ = _gen_float_intrinsic('fabs', 1)
floor = _gen_float_intrinsic('floor', 1)
ceil = _gen_float_intrinsic('ceil', 1)
trunc = _gen_float_intrinsic('trunc', 1)
rint = _gen_float_intrinsic('rint', 1)
nearbyint = _gen_float_intrinsic('nearbyint', 1)
round_ = _gen_float_intrinsic('round', 1)

pow_ = _gen_float_intrinsic('pow', 2)
minnum = _gen_float_intrinsic('minnum', 2)
maxnum = _gen_float_intrinsic('maxnum', 2)
copysign = _gen_float_intrinsic('copysign', 2)

fma = _gen_float_intrinsic('fma', 3)
fmuladd = _gen_float_intrinsic('fmuladd', 3)

#   llvm.powi.*


@abs_.register
def _int_impl(value):

    if UnsignedIntegerType.test_dtype_class_or_vector(value):
        return value
    elif SignedIntegerType.test_dtype_class_or_vector(value):
        return Select(ge(value, 0), value, -value)
    else:
        return NotImplemented


@copysign.register
def _int_impl(mag, sgn):

    try:
        mag, sgn = _convert_python_numbers(mag, sgn)
    except TypeError:
        return NotImplemented
    if not isinstance(mag, Expression) or not isinstance(sgn, Expression):
        return NotImplemented
    if mag.dtype != sgn.dtype:
        return NotImplemented
    if UnsignedIntegerType.test_dtype_class_or_vector(mag):
        return mag
    elif SignedIntegerType.test_dtype_class_or_vector(mag):
        return Select(ge(mag*sgn, 0), mag, -mag)
    else:
        return NotImplemented


def _noop_int(value):

    if IntegerType.test_dtype_class_or_vector(value):
        return value
    else:
        return NotImplemented


floor.register(_noop_int)
ceil.register(_noop_int)
trunc.register(_noop_int)
rint.register(_noop_int)
nearbyint.register(_noop_int)
round_.register(_noop_int)


del _int_impl, _noop_int


# vim: ts=4:sts=4:sw=4:et
