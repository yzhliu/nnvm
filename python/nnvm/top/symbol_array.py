# pylint: disable=invalid-name
"""Symbol array object used by weight pre-pack"""
import tvm

_sym_arr_get = tvm.get_global_func("nnvm.compiler._symbol_array_get")
_sym_arr_size = tvm.get_global_func("nnvm.compiler._symbol_array_size")

class SymbolArray(object):
    """Symbol array in nnvm.

    Used by python registration of alter_op_layout function.
    SymbolArray is passed as the second argument to alter_op_layout function.
    """
    _tvm_tcode = 19

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        tvm.nd.free_extension_handle(self.handle, 19)

    @property
    def _tvm_handle(self):
        return self.handle.value

    def __getitem__(self, key):
        return _sym_arr_get(self, key)

    def __len__(self):
        return _sym_arr_size(self)

    def __repr__(self):
        return str([self[i] for i in len(self)])

tvm.register_extension(SymbolArray, SymbolArray)
