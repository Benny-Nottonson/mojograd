alias ADD_ID = 1
alias MUL_ID = 2
alias POW_ID = 3
alias RELU_ID = 4
alias SUB_ID = 5

@register_passable("trivial")
struct Value[T: DType]:
    var data: Scalar[T]
    var grad: Scalar[T]
    var l: Pointer[Value[T]]
    var r: Pointer[Value[T]]
    var op: Int

    alias null = Pointer[Value[T]].get_null()

    @always_inline("nodebug")
    fn __init__(inout self, data: Scalar[T]):
        self.data = data
        self.grad = 0
        self.l = Self.null
        self.r = Self.null
        self.op = 0

    @always_inline("nodebug")
    fn __add__(self, inout other: Pointer[Value[T]]) -> Value[T]:
        var new_value = Self(self.data + other[].data)

        new_value.l = Pointer[Value[T]].alloc(1)
        new_value.l[] = self
        new_value.r = other
        new_value.op = ADD_ID

        return new_value

    @always_inline("nodebug")
    fn __add__(self, inout other: Scalar[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = Self(other)
        return self.__add__(ptr)

    @always_inline("nodebug")
    fn __add__(self, inout other: Value[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = other
        return self.__add__(ptr)

    @always_inline("nodebug")
    fn __sub__(self, inout other: Pointer[Value[T]]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = -other[]
        return self.__add__(ptr)

    @always_inline("nodebug")
    fn __sub__(self, inout other: Scalar[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = Self(other)
        return self.__sub__(ptr)

    @always_inline("nodebug")
    fn __sub__(self, inout other: Value[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = other
        return self.__sub__(ptr)

    @always_inline("nodebug")
    fn __mul__(self, inout other: Pointer[Value[T]]) -> Value[T]:
        var new_value = Self(self.data * other[].data)

        new_value.l = Pointer[Value[T]].alloc(1)
        new_value.l[] = self
        new_value.r = other
        new_value.op = MUL_ID

        return new_value

    @always_inline("nodebug")
    fn __mul__(self, inout other: Scalar[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = Self(other)
        return self.__mul__(ptr)

    @always_inline("nodebug")
    fn __mul__(self, inout other: Value[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = other
        return self.__mul__(ptr)

    @always_inline("nodebug")
    fn __pow__(self, inout other: Pointer[Value[T]]) -> Value[T]:
        var new_value = Self(self.data ** other[].data)

        new_value.l = Pointer[Value[T]].alloc(1)
        new_value.l[] = self
        new_value.r = other
        new_value.op = POW_ID

        return new_value

    @always_inline("nodebug")
    fn __pow__(self, inout other: Scalar[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = Self(other)
        return self.__pow__(ptr)

    @always_inline("nodebug")
    fn __pow__(self, inout other: Value[T]) -> Value[T]:
        var ptr = Pointer[Value[T]].alloc(1)
        ptr[] = other
        return self.__pow__(ptr)

    @always_inline("nodebug")
    fn relu(self) -> Value[T]:
        var new_value = Value[T](0 if self.data < 0 else self.data)

        new_value.l = Pointer[Value[T]].alloc(1)
        new_value.l[] = self
        new_value.op = RELU_ID

        return new_value

    @always_inline("nodebug")
    fn __neg__(self) -> Value[T]:
        var data: Scalar[T] = -1
        return self * data

    @always_inline("nodebug")
    fn __radd__(self, inout other: Value[T]) -> Value[T]:
        return self + other

    @always_inline("nodebug")
    fn __radd__(self, inout other: Scalar[T]) -> Value[T]:
        return self + other

    @always_inline("nodebug")
    fn __rsub__(self, inout other: Pointer[Value[T]]) -> Value[T]:
        return self - other

    @always_inline("nodebug")
    fn __rsub__(self, inout other: Scalar[T]) -> Value[T]:
        return self - other

    @always_inline("nodebug")
    fn __rmul__(self, inout other: Pointer[Value[T]]) -> Value[T]:
        return self * other

    @always_inline("nodebug")
    fn __rmul__(self, inout other: Scalar[T]) -> Value[T]:
        return self * other

    @always_inline("nodebug")
    fn __truediv__(self, inout other: Value[T]) -> Value[T]:
        var neg_one: Scalar[T] = -1.0
        var pow = other ** neg_one
        return self * pow

    @always_inline("nodebug")
    fn __rtruediv__(self, inout other: Value[T]) -> Value[T]:
        var powie: Scalar[T] = -1.0
        var pow = self ** powie
        return other * pow

    @always_inline("nodebug")
    fn __truediv__(self, inout other: Scalar[T]) -> Value[T]:
        var o = Value[T](other)
        return self / o

    @always_inline("nodebug")
    fn __rtruediv__(self, inout other: Scalar[T]) -> Value[T]:
        var o = Value[T](other)
        return  o / self

    @staticmethod
    @always_inline("nodebug")
    fn backward(inout v: Pointer[Value[T]]):
        var op = v[].op

        if op == 0:
            return
        elif op == ADD_ID:
            Self.backward_add(v)
        elif op == MUL_ID:
            Self.backward_mul(v)
        elif op == POW_ID:
            Self.backward_pow(v)
        elif op == RELU_ID:
            Self.backward_relu(v)
        else:
            print("Operation not supported")

    @staticmethod
    @always_inline("nodebug")
    fn backward_add(inout v: Pointer[Value[T]]):
        var x = v[]

        if x.l == Self.null:
            return

        x.l[].grad += x.grad

        if x.r == Self.null:
            return

        x.r[].grad += x.grad

    @staticmethod
    @always_inline("nodebug")
    fn backward_mul(inout v: Pointer[Value[T]]):
        var x = v[]

        if x.l == Self.null:
            return

        if x.r == Self.null:
            return

        var l = x.l[]
        var r = x.r[]

        l.grad += r.data * x.grad
        r.grad += l.data * x.grad

    @staticmethod
    @always_inline("nodebug")
    fn backward_pow(inout v: Pointer[Value[T]]):
        var x = v[]

        if x.l == Self.null:
            return

        if x.r == Self.null:
            return

        var l = x.l[]
        var r = x.r[]

        l.grad += r.data * l.data ** (r.data - 1) * x.grad

    @staticmethod
    @always_inline("nodebug")
    fn backward_relu(inout v: Pointer[Value[T]]):
        var x = v[]

        if x.l == Self.null:
            return

        var l = x.l[]

        if x.data > 0:
            l.grad += x.grad

    @staticmethod
    fn build_topo(inout v: Pointer[Value[T]], inout visited: List[Pointer[Value[T]]], inout topo: List[Pointer[Value[T]]]):
        if v == Self.null:
            return

        var x = v[]
        var is_visited = False
        var size = len(visited)

        for i in range(size):
            if v == visited[i]:
                is_visited = True

        if not is_visited:
            visited.append(v)

            if x.l != Self.null:
                Value.build_topo(x.l, visited, topo)

            if x.r != Self.null:
                Value.build_topo(x.r, visited, topo)

            topo.append(v)

    @staticmethod
    fn backward(inout self):
        var visited = List[Pointer[Value[T]]]()
        var topo = List[Pointer[Value[T]]]()
        var ptr_self = Pointer[Value[T]].alloc(1)

        ptr_self[] = self
        self.grad = 1

        Value.build_topo(ptr_self, visited, topo)
        
        for i in range(0, len(topo), -1):
            Value.backward(topo[i])

        visited.clear()
        topo.clear()
        ptr_self.free()

    @always_inline("nodebug")
    fn print(inout self):
        print("<Value", "data:", self.data, "grad:", self.grad, "op:", Self.op_rep(self.op), ">")

    @always_inline("nodebug")
    fn print(inout self, label: StringRef):
        print("<Value", "label:", label, "data:", self.data, "grad:", self.grad, "op:", Self.op_rep(self.op), ">")

    @staticmethod
    @always_inline("nodebug")
    fn op_rep(op: Int) -> String:
        if op == ADD_ID:
            return "Add"
        elif op == MUL_ID:
            return "Mul"
        elif op == SUB_ID:
            return "Sub"
        elif op == RELU_ID:
            return "Relu"
        elif op == POW_ID:
            return "Pow"
        else:
            return "Unkown"