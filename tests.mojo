from memory.unsafe import Pointer
from python import Python

from src import Value, Neuron, Layer, MLP, make_moons, plot, print_datasets

alias T = DType.float32

fn plot_classifier_step(step: Int, inout model: MLP[T], inout X: Pointer[Pointer[Value[T]]], n_samples: Int) raises:
    var outs = Pointer[Pointer[Value[T]]].alloc(n_samples)

    for i in range(n_samples):
        var x = X.load(i)
        var packed = model.forward(x)
        var zero_or_one = 0 if packed[0].data > 0 else 1
        var v = Value[T](zero_or_one)
        var ptr_v = Pointer[Value[T]].alloc(1)
        ptr_v[] = v

        outs.store(i, ptr_v)

fn asset_input(n_x: Int = 2) -> Pointer[Value[T]]:
    var x = Pointer[Value[T]].alloc(n_x)

    for i in range(n_x):
        x[i] = Value[T](2.0)

    return x


fn test_simple_eq():
    var a = Value[T](2.0)
    var b = Value[T](3.0)
    var c: Scalar[T] = 2.0
    var d = b ** c
    var e = a + c

    e.backward(e)

    a.print()
    b.print()
    d.print()
    e.print()

fn test_autograd():
    var a = Value[T](2.0)
    var b = Value[T](1.0)
    var c = Value[T](4.0)
    var x = a + -b
    var x2 = c + c
    var x3 = x + x2
    var x4 = x3.relu()

    print(a.data, b.data, c.data, x4.data)
    print(a.grad, b.grad, c.grad, x4.grad)
    x4.backward(x4)
    print(a.data, b.data, c.data, x4.data)
    print(a.grad, b.grad, c.grad, x4.grad)



fn test_neuron():
    var x = asset_input(2)

    var neuron = Neuron[T](2)
    var ptr_s = neuron.forward(x)
    print("s", ptr_s.data, ptr_s.grad)
    var s = ptr_s
    s.backward(s)
    print("s", ptr_s.data, ptr_s.grad)
    
    for i in range(neuron.inputs):
        var v = neuron.weights[i]
        print("w", i, v.data, v.grad)


fn test_layer():
    var x = asset_input(2)

    var l = Layer[T](2, 1)
    var res = l.forward(x)

    for i in range(l.outputs):
        var v = res[i]
        print("v", i, v.data, v.grad)
        v.backward(v)
        print("v", i, v.data, v.grad)


fn test_mlp() raises:
    var x = asset_input()

    var nouts = List[Int]()
    nouts.append(16)
    nouts.append(16)
    nouts.append(1)

    var m = MLP[T](2, nouts)

    var res = m.forward(x)

    var res_v = res[0]
    print("v", res_v.data, res_v.grad)
    res_v.backward(res_v)
    print("v", res_v.data, res_v.grad)


fn test_optmization() raises:
    var time = Python.import_module("time")

    var n_samples = 30
    var n_dim = 2
    var out = make_moons[T](n_samples, 0.1)
    var X = out.get[0, Pointer[Pointer[Value[T]]]]()
    var y = out.get[1, Pointer[Value[T]]]()

    var nouts = List[Int]()
    nouts.append(16)
    nouts.append(16)
    nouts.append(1)

    var model = MLP[T](2, nouts)

    var num_epochs = 100
    var scores = Pointer[Pointer[Value[T]]].alloc(n_samples)
    var losses = Pointer[Pointer[Value[T]]].alloc(n_samples)
    for i in range(n_samples):
        var ptr_loss = Pointer[Value[T]].alloc(1)
        losses.store(i, ptr_loss)

    for k in range(num_epochs):
        var scores = Pointer[Value[T]].alloc(n_samples)

        for i in range(n_samples):
            var val = X[i]
            scores[i] = model.forward(X[i])[]

        for i in range(n_samples):
            var scorei = scores[i]
            var yi = y[i]
            var prod = -yi * scorei
            var one: Scalar[T] = 1.0
            var loss = (one + prod).relu()
            var ptr_loss = losses.load(i)
            ptr_loss.store(loss)

        var sum_losses = Value[T](0.0)
        for i in range(n_samples):
            sum_losses = sum_losses + losses[i][]
        var div: Float32 = 1.0 / n_samples
        var data_loss = sum_losses * div

        model.zero_grad()
        data_loss.backward(data_loss)

        var params = model.parameters()
        var learning_rate = 1.0 - 0.9*k/100
        for i in range(len(params)):
            var param = params[i]
            var data = param.data
            data -= learning_rate * param.grad

        print("step", k, "loss", data_loss.data)

        var outs = Pointer[Pointer[Value[T]]].alloc(n_samples)

        for i in range(n_samples):
            var x = X.load(i)
            var packed = model.forward(x)
            var zero_or_one = 0 if packed[0].data > 0 else 1
            var v = Value[T](zero_or_one)
            var ptr_v = Pointer[Value[T]].alloc(1)
            ptr_v.store(v)
            outs[i] = ptr_v

        plot_classifier_step(k, model, X, n_samples)

fn main() raises:
    test_simple_eq()
    test_autograd()
    test_neuron()
    test_layer()
    test_mlp()
    test_optmization()