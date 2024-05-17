from random import random_float64

@register_passable("trivial")
struct Neuron[T: DType]:
    var weights: Pointer[Value[T]]
    var bias: Value[T]
    var inputs: Int
    var linear: Bool

    @always_inline("nodebug")
    fn __init__(inout self, inputs: Int, linear: Bool = False):
        var weights = Pointer[Value[T]].alloc(inputs)

        for i in range(inputs):
            weights[i] = Value[T](random_float64(-1, 1))

        self.weights = weights
        self.bias = Value[T](0)
        self.inputs = inputs
        self.linear = linear

    @always_inline("nodebug")
    fn forward(inout self, inout x: Pointer[Value[T]]) -> Value[T]:
        var total = Value[T](0)

        for i in range(self.inputs):
            var s = self.weights[i] * x[i]
            total = total + s

        var act = total + self.bias

        if self.linear:
            return act
        
        return act.relu()

    @always_inline("nodebug")
    fn parameters(self) -> List[Value[T]]:
        var params = List[Value[T]]()

        for i in range(self.inputs):
            params.append(self.weights[i])

        params.append(self.bias)

        return params

@register_passable("trivial")
struct Layer[T: DType]:
    var neurons: Pointer[Neuron[T]]
    var inputs: Int
    var outputs: Int
    var linear: Bool

    @always_inline("nodebug")
    fn __init__(inout self, inputs: Int, outputs: Int, linear: Bool = False):
        var neurons = Pointer[Neuron[T]].alloc(outputs)

        for i in range(outputs):
            neurons[i] = Neuron[T](inputs, linear)

        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.linear = linear

    @always_inline("nodebug")
    fn forward(inout self, inout x: Pointer[Value[T]]) -> Pointer[Value[T]]:
        var out = Pointer[Value[T]].alloc(self.outputs)

        for i in range(self.outputs):
            out[i] = self.neurons[i].forward(x)

        return out

    @always_inline("nodebug")
    fn parameters(self) -> List[Value[T]]:
        var params = List[Value[T]]()

        for i in range(self.outputs):
            var n_params = self.neurons[i].parameters()

            for j in range(len(n_params)):
                params.append(n_params[j])

        return params

@register_passable("trivial")
struct MLP[T: DType]:
    var layers: Pointer[Layer[T]]
    var nlayers: Int

    @always_inline("nodebug")
    fn __init__(inout self, inputs: Int, outputs: List[Int]):
        var nnodes = len(outputs) + 1
        var nlayers = nnodes - 1
        var layers = Pointer[Layer[T]].alloc(nlayers)

        layers[0] = Layer[T](inputs, outputs[0], True)

        for i in range(nlayers - 1):
            layers[i + 1] = Layer[T](outputs[i], outputs[i + 1], i != nlayers - 2)

        self.layers = layers
        self.nlayers = nlayers

    @always_inline("nodebug")
    fn forward(inout self, inout x: Pointer[Value[T]]) -> Pointer[Value[T]]:
        for i in range(self.nlayers):
            x = self.layers[i].forward(x)
        return x

    @always_inline("nodebug")
    fn parameters(self) -> List[Value[T]]:
        var params = List[Value[T]]()

        for i in range(self.nlayers):
            var layer_params = self.layers[i].parameters()

            for j in range(len(layer_params)):
                params.append(layer_params[j])

        return params

    @always_inline("nodebug")
    fn zero_grad(inout self):
        var params = self.parameters()

        for i in range(len(params)):
            params[i].grad = 0