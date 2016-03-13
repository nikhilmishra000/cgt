from __future__ import division
import numpy as np
import cgt
from cgt import nn
import types
import cPickle as pickle
from matplotlib import pyplot as plt

def make_variable(name, shape):
    return "{0} = cgt.tensor(cgt.floatX, {2}, fixed_shape={1})".format(name, shape, len(shape))

def normalize(var):
    return cgt.broadcast("/", var, cgt.sum(var,axis=2,keepdims=True), "xxx,xx1")

class struct(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

class Params(object):
    def __init__(self,params):
        assert all(param.is_data() and param.dtype == cgt.floatX for param in params)
        self._params = params

    @property
    def params(self):
        return self._params

    def get_values(self):
        return [param.op.get_value() for param in self._params]

    def get_shapes(self):
        return [param.op.get_shape() for param in self._params]

    def get_total_size(self):
        return sum(np.prod(shape) for shape in self.get_shapes())

    def num_vars(self):
        return len(self._params)

    def set_values(self, parvals):
        assert len(parvals) == len(self._params)
        for (param, newval) in zip(self._params, parvals):
            param.op.set_value(newval)
            param.op.get_shape() == newval.shape

    def set_value_flat(self, theta):
        theta = theta.astype(cgt.floatX)
        arrs = []
        n = 0        
        for shape in self.get_shapes():
            size = np.prod(shape)
            arrs.append(theta[n:n+size].reshape(shape))
            n += size
        assert theta.size == n
        self.set_values(arrs)
    
    def get_value_flat(self):
        theta = np.empty(self.get_total_size(),dtype=cgt.floatX)
        n = 0
        for param in self._params:
            s = param.op.get_size()
            theta[n:n+s] = param.op.get_value().flat
            n += s
        assert theta.size == n
        return theta


def backtrack(func, x, dx, f, g, alpha=0.25, beta=0.9, lb=1e-6, ub=1):
    t = ub
    while func(x + t*dx) >= f + t*alpha*g.dot(dx):
        t *= beta
        if t < lb:
            print "Warning: backtracking hit lower bound of {0}".format(lb)
            break;
    return beta
    
class Solver(struct):
    def __init__(self, args):
        """
        @param args, a dict containing:
        paramters: alpha, momentum, lr_decay, lr_step, min_alpha
        rmsp: gamma
        etc: plot_step, plot_func, fname, snapstep
        """
        struct.__init__(self, **args)
        if self.plot_step:
            self.plot = plt.figure(); plt.ion(); plt.show(0)
            
    def initialize(self, model):
        self.model = model
        self.dtheta = np.zeros_like(model.theta)
        self._iter, self.loss = 0, [[]]

    def check_nan(self, loss, grad):
        return np.isnan(loss) or np.isnan(np.linalg.norm(grad))
        
    def decay_lr(self):
        if self._iter % self.lr_step == 0:
            self.alpha *= self.lr_decay
            self.alpha = max(self.alpha, self.min_alpha)
            
    def draw(self):
        if self.plot_step and self._iter % self.plot_step == 0:
            self.plot.clear()
            self.loss[-1] = np.mean(self.loss[-1])
            self.plot_func(range(0,self._iter,self.plot_step), self.loss)
            self.loss.append([])
            plt.draw()

    def snapshot(self):
        if len(self.fname) > 0 and self._iter % self.snap_step == 0:
            self.model.dump(self.fname.format(self._iter),
                            {k:v for k,v in self.items() if type(v) != types.FunctionType and not isinstance(v, Model)})
        
    def update(self, loss, grad, acc, disp):
        if self.check_nan(loss, grad):
            if self.pass_nan:
                print "something is nan, skipping..."
                return np.zeros_like(self.dtheta),False
            else:
                return np.zeros_like(self.dtheta),True
            
        self._iter += 1; self.loss[-1].append(float(loss))
        if acc is not None and disp:
            print "iter: {:0d}, loss: {:1.4e}, gnorm: {:2.4e}, alpha: {:3.4e}, acc: {:4.4f}"\
                .format(self._iter, float(loss), np.abs(grad).max(), self.alpha, acc)
        elif disp:
            print "iter: {:0d}, loss: {:1.4e}, gnorm: {:2.4e}, alpha: {:3.4e}"\
                .format(self._iter, float(loss), np.abs(grad).max(), self.alpha)
                
        self.decay_lr()
        self.draw()
        self.snapshot()
        return self.dtheta,False

class RMSP(Solver):
    def initialize(self, theta):
        Solver.initialize(self, theta)
        self.sqgrad = np.zeros_like(theta) + 1e-6
        self.method = "rmsp"
        
    def update(self, loss, grad, acc=None, disp=True):
        self.sqgrad = self.gamma*self.sqgrad + (1-self.gamma)*np.square(grad)
        self.dtheta = self.momentum * self.dtheta - \
                      (1-self.momentum) * self.alpha * grad / np.sqrt(self.sqgrad)
        return Solver.update(self, loss, grad, acc, disp)

class CG(Solver):
    def initialize(self, theta):
        Solver.initialize(self, theta)
        self.prev_grad = np.zeros_like(theta)
        self.method = 'cg'

    def update(self, loss, grad, acc=None, disp=True):
        dx,dxp = -grad, self.prev_grad
        beta = max(0, dx.dot(dx - dxp) / (dxp.dot(dxp)-1e-10) )
        self.dtheta = dx + beta*dxp
        
        if self.allow_backtrack:
            self.alpha = backtrack(self.model.func, self.model.theta, self.dtheta, loss, grad)
            
        self.dtheta *= self.alpha
        self.prev_grad = dx
        return Solver.update(self, loss, grad, acc, disp)
"""
not yet implemented:

class LBFGS(Solver):
    def initialize(self, theta):
        Solver.initialize(self, theta)
        self.method = "lbfgs"

    def update(self, loss, grad, acc):
        q = grad; s,y = self.s, self.y
        for i in range(m):
            q  -= 0
        return Solver.update(self, loss, grad, acc)
"""

class Model(object):
    def initialize(self, loss, scale):
        self._iter = 0
        self.pc = Params(self.params)
        cur_val = self.pc.get_value_flat()
        idx = cur_val.nonzero()
        new_val = np.random.uniform(-scale, scale, size=(self.pc.get_total_size(),))
        new_val[idx] = cur_val[idx]
        self.sync(new_val)
        grad = cgt.concatenate([g.flatten() for g in cgt.grad(loss, self.params)])
        return grad
    
    def sync(self, theta):
        self.theta = theta.copy()
        self.pc.set_value_flat(self.theta)

    def dump(self, fname, args={}):
        with open(fname, 'wb') as f:
            args["theta"], args["shapes"] = self.theta, self.pc.get_shapes()
            pickle.dump(args, f)

    def restore(self, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
            self.sync(data["theta"])
            return data
            
class TextData(object):
    def __init__(self, input_file):
        with open(input_file) as f:
            raw = f.read()
            f.close()            
        self._raw = raw
        self._encoding, self._decoding = {}, {}
        for ch in self._raw:
            if ch not in self._encoding:
                self._encoding[ch] = len(self._encoding)
                self._decoding[self._encoding[ch]] = ch
        self._heads = []
        
    @property
    def size(self):
        return len(self._encoding)

    def encode(self, ch):
        return np.array([[1 if i == self._encoding[ch] else 0 for i in xrange(self.size)]])

    def decode(self, num):
        return self._decoding[num]

    def get_batch(self, seq_length, batch_size):
        seq_length += 1; batch = np.zeros((batch_size, seq_length, self.size), dtype=np.float32)
        self._heads = [i * len(self._raw)//batch_size for i in range(batch_size)]
        
        for b in range(batch_size):
            seg = self._raw[self._heads[b]:self._heads[b] + seq_length]
            batch[b] = np.array([self.encode(ch).ravel() for ch in seg])
            self._heads[b] = (self._heads[b] + seq_length) % (len(self._raw) - seq_length)
        batch = batch.transpose((1,0,2))
        return batch[:-1], batch[1:]
