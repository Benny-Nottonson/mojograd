from python import Python
from random import random_float64

@always_inline("nodebug")
fn make_moons[T: DType](n_samples: Int, noise: Float64) raises -> (Pointer[Pointer[Value[T]]], Pointer[Value[T]]):
    var sklearn = Python.import_module("sklearn.datasets")
    var numpy = Python.import_module("numpy")
    var out = sklearn.make_moons(n_samples)
    var py_X = out[0]
    var py_y = out[1]

    var X = Pointer[Pointer[Value[T]]].alloc(n_samples)

    for i in range(n_samples):
        var row = Pointer[Value[T]].alloc(2)
        for j in range(2):
            var v = py_X[i][j].to_float64()
            var noise = random_float64(-noise, noise)
            row[j] = Value[T](v + noise)
        X[i] = row

    var y = Pointer[Value[T]].alloc(n_samples)
    
    for i in range(n_samples):
        var v = py_y[i].to_float64()
        y[i] = Value[T](v * 2 - 1)

    return X, y

@always_inline("nodebug")
fn print_datasets[T: DType](X: Pointer[Pointer[Value[T]]], y: Pointer[Value[T]], n_samples: Int):
    print("X:")
    for i in range(n_samples):
        var row = X[i]
        print(i, row[0].data, row[1].data)

    print("y:")
    for i in range(n_samples):
        print(i, y[i].data)

@always_inline("nodebug")
fn plot[T: DType](X: Pointer[Pointer[Value[T]]], y: Pointer[Value[T]], n_samples: Int, filename: String, title: String) raises:
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")
    var x0 = np.zeros(n_samples, np.float32)
    var x1 = np.zeros(n_samples, np.float32)
    var yy = np.zeros(n_samples, np.float32)

    for i in range(n_samples):
        _ = x0.itemset(i, X[i][0].data)
        _ = x1.itemset(i, X[i][1].data)
        _ = yy.itemset(i, y[i].data + 100)
    
    _ = plt.title(title)
    _ = plt.scatter(x0, x1, 10, yy)
    _ = plt.savefig(filename)