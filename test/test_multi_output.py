import cgt, numpy as np
import unittest


class SinCos(cgt.Op):
#     def c_code(self, inputs):
#         return """
# void CGT_FUNCNAME(void* cldata, cgt_array** io) {
#     float* x = io[0]->data;
#     float* y = io[1]->data;
#     float* z = io[2]->data;
#     y[0] = sinf(x[0]);
#     z[0] = cosf(x[0]);
# }
#         """
    call_type = "valret"
    def typ_apply(self, inputs):
        d = inputs[0].ndim
        return cgt.Tuple(cgt.Tensor(cgt.floatX, d), cgt.Tensor(cgt.floatX, d))
    def py_apply_valret(self, reads):
        x = reads[0]
        return (np.sin(x), np.cos(x))
    def shp_apply(self, inputs):
        return (cgt.shape(inputs[0]), cgt.shape(inputs[0]))
    c_extra_link_flags = "-lm"
    c_extra_includes = ["math.h"]

class SinCos2(cgt.Op):
    def c_code(self, inputs):
        # raise cgt.MethodNotDefined
        return """
using namespace cgt;
extern "C" void CGT_FUNCNAME(void* cldata, Array** reads, Tuple* write) {
    float* x = static_cast<float*>(reads[0]->data);
    float* y = static_cast<float*>(static_cast<Array*>(write->getitem(0))->data);
    float* z = static_cast<float*>(static_cast<Array*>(write->getitem(1))->data);
    for (int i=0; i < cgt_size(reads[0]); ++i) {
        y[i] = sinf(x[i]);
        z[i] = cosf(x[i]);    
    }
}
        """
    call_type = "inplace"
    def typ_apply(self, inputs):
        ndim = inputs[0].ndim
        return cgt.Tuple(cgt.Tensor(cgt.floatX, ndim), cgt.Tensor(cgt.floatX, ndim))
    def py_apply_inplace(self, reads, write):
        x = reads[0]
        write[0][...] = np.sin(x)
        write[1][...] = np.cos(x)
    def shp_apply(self, inputs):
        return (cgt.shape(inputs[0]), cgt.shape(inputs[0]))
    c_extra_link_flags = "-lm"
    c_extra_includes = ["math.h"]

class MultiOutputTestCase(unittest.TestCase):
    def runTest(self):
        for x in (cgt.scalar('x'), cgt.vector('x'), cgt.matrix('x')):
            for cls in (SinCos, SinCos2):
                y,z = cgt.unpack(cgt.Result(cls(), [x]))
                xnum = np.ones((3,)*x.ndim, cgt.floatX)
                correct = (np.sin(xnum),np.cos(xnum))
                yznum = cgt.numeric_eval([y,z], {x:xnum})
                np.testing.assert_allclose(yznum, correct)
                f = cgt.make_function([x],[y,z])
                np.testing.assert_allclose(f(xnum), correct)


if __name__ == "__main__":
    MultiOutputTestCase().runTest()
