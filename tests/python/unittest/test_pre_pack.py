import nnvm.symbol as sym
from nnvm.compiler import graph_util
from nnvm import graph
from nnvm.top import registry as reg

@reg.register_weight_prepack("relu")
def weight_prepack_conv2d(attrs, inputs, tinfos):
    print("Relu attrs : " + str(attrs))
    print("Relu input size: " + str(len(inputs)))
    print("Relu input 0: " + str(inputs[0]))
    print("Relu tinfos size: " + str(len(tinfos)))
    print("Relu tinfos[0]: " + str(tinfos))
    s = sym.relu(data=inputs[0], name='relu_preprocess')
    return sym.softmax(data=s, name='softmax_preprocess')
    # return inputs[0]

def test_pre_pack():
    h = 128
    w = 128
    data_shape = (1, 3, h, w)
    data = sym.Variable('data', shape=data_shape, dtype=0)
    relu = sym.relu(data=data, name='relu', test_attr='hahaha')
    softmax = sym.softmax(data=relu, name='softmax')

    g = graph.create(softmax) 
    g.apply("InferShape").apply("InferType").apply("PrePack")

    print(g.json())
    """
    with open('graph.json', 'w') as fn:
        fn.writelines(g.json())
    """

if __name__ == "__main__":
    test_pre_pack()
