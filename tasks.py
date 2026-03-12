
import numpy as np
import torch

class TempOrderTask:
    """Temporal Order task (2 bits -> 4 classes), lastSoftmax."""
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.nin = 6
        self.nout = 4
        self.classifType = "lastSoftmax"
        self.report = "last"

    def generate(self, batchsize: int, length: int):
        l = length
        p0 = self.rng.randint(int(l * 0.1), size=(batchsize,)) + int(l * 0.1)
        v0 = self.rng.randint(2, size=(batchsize,))
        p1 = self.rng.randint(int(l * 0.1), size=(batchsize,)) + int(l * 0.5)
        v1 = self.rng.randint(2, size=(batchsize,))
        targ_vals = v0 + v1 * 2
        vals = self.rng.randint(4, size=(l, batchsize)) + 2
        vals[p0, np.arange(batchsize)] = v0
        vals[p1, np.arange(batchsize)] = v1
        data = np.zeros((l, batchsize, 6), dtype=np.float32)
        targ = np.zeros((batchsize, 4), dtype=np.float32)
        data.reshape((l * batchsize, 6))[np.arange(l * batchsize), vals.flatten()] = 1.0
        targ[np.arange(batchsize), targ_vals] = 1.0
        return data, targ

class TempOrder3bitTask:
    """Temporal Order 3-bit task (3 bits -> 8 classes), lastSoftmax."""
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.nin = 6
        self.nout = 8
        self.classifType = "lastSoftmax"
        self.report = "last"

    def generate(self, batchsize: int, length: int):
        l = length
        p0 = self.rng.randint(int(l * 0.1), size=(batchsize,)) + int(l * 0.1)
        v0 = self.rng.randint(2, size=(batchsize,))
        p1 = self.rng.randint(int(l * 0.1), size=(batchsize,)) + int(l * 0.3)
        v1 = self.rng.randint(2, size=(batchsize,))
        p2 = self.rng.randint(int(l * 0.1), size=(batchsize,)) + int(l * 0.6)
        v2 = self.rng.randint(2, size=(batchsize,))
        targ_vals = v0 + v1 * 2 + v2 * 4
        vals = self.rng.randint(4, size=(l, batchsize)) + 2
        vals[p0, np.arange(batchsize)] = v0
        vals[p1, np.arange(batchsize)] = v1
        vals[p2, np.arange(batchsize)] = v2
        data = np.zeros((l, batchsize, 6), dtype=np.float32)
        targ = np.zeros((batchsize, 8), dtype=np.float32)
        data.reshape((l * batchsize, 6))[np.arange(l * batchsize), vals.flatten()] = 1.0
        targ[np.arange(batchsize), targ_vals] = 1.0
        return data, targ

class AddTask:
    """Adding task (two marked positions); lastLinear regression; target is mean of two values."""
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.nin = 2
        self.nout = 1
        self.classifType = "lastLinear"
        self.report = "last"

    def generate(self, batchsize: int, length: int):
        l = self.rng.randint(int(length * 0.1)) + length
        p0 = self.rng.randint(int(l * 0.1), size=(batchsize,))
        p1 = self.rng.randint(int(l * 0.4), size=(batchsize,)) + int(l * 0.1)
        data = self.rng.uniform(size=(l, batchsize, 2)).astype(np.float32)
        data[:, :, 0] = 0.0
        data[p0, np.arange(batchsize), np.zeros((batchsize,), dtype=np.int32)] = 1.0
        data[p1, np.arange(batchsize), np.zeros((batchsize,), dtype=np.int32)] = 1.0
        targs = (data[p0, np.arange(batchsize), np.ones((batchsize,), dtype=np.int32)] +
                 data[p1, np.arange(batchsize), np.ones((batchsize,), dtype=np.int32)]) / 2.0
        return data, targs.reshape((-1, 1)).astype(np.float32)

class MulTask:
    """Multiplication task (two marked positions); lastLinear regression; target is product of two values."""
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.nin = 2
        self.nout = 1
        self.classifType = "lastLinear"
        self.report = "last"

    def generate(self, batchsize: int, length: int):
        l = self.rng.randint(int(length * 0.1)) + length
        p0 = self.rng.randint(int(l * 0.1), size=(batchsize,))
        p1 = self.rng.randint(int(l * 0.4), size=(batchsize,)) + int(l * 0.1)
        data = self.rng.uniform(size=(l, batchsize, 2)).astype(np.float32)
        data[:, :, 0] = 0.0
        data[p0, np.arange(batchsize), np.zeros((batchsize,), dtype=np.int32)] = 1.0
        data[p1, np.arange(batchsize), np.zeros((batchsize,), dtype=np.int32)] = 1.0
        targs = (data[p0, np.arange(batchsize), np.ones((batchsize,), dtype=np.int32)] *
                 data[p1, np.arange(batchsize), np.ones((batchsize,), dtype=np.int32)])
        return data, targs.reshape((-1, 1)).astype(np.float32)

class PermTask:
    """Permutation task; lastSoftmax over 100 classes."""
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.nin = 100
        self.nout = 100
        self.classifType = "lastSoftmax"
        self.report = "last"

    def generate(self, batchsize: int, length: int):
        randvals = self.rng.randint(98, size=(length + 1, batchsize)) + 2
        val = self.rng.randint(2, size=(batchsize,))
        randvals[np.zeros((batchsize,), dtype=np.int32), np.arange(batchsize)] = val
        randvals[np.ones((batchsize,), dtype=np.int32) * length, np.arange(batchsize)] = val
        _targ = randvals[1:]
        _inp = randvals[:-1]
        inp = np.zeros((length, batchsize, 100), dtype=np.float32)
        targ = np.zeros((1, batchsize, 100), dtype=np.float32)
        inp.reshape((length * batchsize, 100))[np.arange(length * batchsize), _inp.flatten()] = 1.0
        targ.reshape((batchsize, 100))[np.arange(batchsize), _targ[-1].flatten()] = 1.0
        return inp, targ.reshape((batchsize, 100)).astype(np.float32)

class MemTask:
    """Memorization task; per-timestep softmax; report='all' (any timestep wrong counts as error)."""
    def __init__(self, rng: np.random.RandomState, n_values: int = 5, n_pos: int = 10, generate_all: bool = False):
        self.rng = rng
        self.n_values = n_values
        self.n_pos = n_pos
        self.dim = (n_values ** n_pos)
        self.generate_all = generate_all

        if generate_all:
            self.data = np.zeros((n_pos, self.dim, n_values + 2), dtype=np.float32)
            for val in range(self.dim):
                tmp_val = val
                for k in range(n_pos):
                    self.data[k, val, tmp_val % n_values] = 1.0
                    tmp_val = tmp_val // n_values

        self.nin = n_values + 2
        self.nout = n_values + 1
        self.classifType = "softmax"
        self.report = "all"

    def generate(self, batchsize: int, length: int):
        if self.generate_all:
            batchsize = self.dim

        input_data = np.zeros((length + 2 * self.n_pos, batchsize, self.n_values + 2), dtype=np.float32)
        targ_data = np.zeros((length + 2 * self.n_pos, batchsize, self.n_values + 1), dtype=np.float32)

        # Default target: "blank" class (last index)
        targ_data[:-self.n_pos, :, -1] = 1.0

        # Input markers
        input_data[self.n_pos:, :, -2] = 1.0
        input_data[length + self.n_pos, :, -2] = 0.0
        input_data[length + self.n_pos, :, -1] = 1.0

        if not self.generate_all:
            self.data = np.zeros((self.n_pos, batchsize, self.n_values + 2), dtype=np.float32)
            for val in range(batchsize):
                tmp_val = self.rng.randint(self.dim)
                for k in range(self.n_pos):
                    self.data[k, val, tmp_val % self.n_values] = 1.0
                    tmp_val = tmp_val // self.n_values

        input_data[:self.n_pos, :, :] = self.data
        targ_data[-self.n_pos:, :, :] = self.data[:, :, :-1]  # predict the stored symbols

        flat_targ = targ_data.reshape(((length + 2 * self.n_pos) * batchsize, -1))
        return input_data, flat_targ.astype(np.float32)


TASKS = {
    "torder": TempOrderTask,
    "torder3": TempOrder3bitTask,
    "add": AddTask,
    "mul": MulTask,
    "perm": PermTask,
    "mem": MemTask,
}

def make_task(name: str, rng: np.random.RandomState, **kwargs):
    if name not in TASKS:
        raise ValueError(f"Unknown task {name}. Choose from {list(TASKS.keys())}")
    cls = TASKS[name]
    if name == "mem":
        return cls(rng, **kwargs)
    return cls(rng)

def to_torch(x_np: np.ndarray, device: str):
    return torch.from_numpy(x_np).to(device=device)

