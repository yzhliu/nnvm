#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/pass_functions.h>
#include "../compiler/graph_transform.h"
#include <tvm/tvm.h>

namespace nnvm {
namespace pass {
namespace {

// convert from type flag to tvm type.
tvm::Type GetTVMType(int type_flag) {
  switch (type_flag) {
    case 0:
      return tvm::Float(32);
    case 1:
      return tvm::Float(64);
    case 2:
      return tvm::Float(16);
    case 3:
      return tvm::UInt(8);
    case 4:
      return tvm::Int(32);
    case 5:
      return tvm::Int(8);
    case 6:
      return tvm::Int(64);
    case 7:
      return tvm::Int(16);
    case 8:
      return tvm::UInt(16);
    case 9:
      return tvm::UInt(32);
    case 10:
      return tvm::UInt(64);
    default:
      LOG(FATAL) << "unknown type_flag=" << type_flag;
      return tvm::Float(32);
  }
}

tvm::Array<tvm::Tensor> GetTensorInfo(const IndexedGraph& idx_graph,
                                      const uint32_t nid,
                                      const ShapeVector& shape_vec,
                                      const DTypeVector& dtype_vec) {
  tvm::Array<tvm::Tensor> vec;
  for (uint32_t i = 0; i < idx_graph[nid].source->num_outputs(); ++i) {
    tvm::Array<tvm::Expr> shape;
    for (int64_t x : shape_vec[idx_graph.entry_id(nid, i)]) {
      CHECK_LE(x, static_cast<int64_t>(std::numeric_limits<int>::max()));
      shape.push_back(tvm::make_const(tvm::Int(32), x));
    }
    vec.push_back(tvm::placeholder(shape, GetTVMType(dtype_vec[idx_graph.entry_id(nid, i)])));
  }
  return vec;
}

Graph PrePack(const Graph& src) {
  static auto& fweight_prepack =
    Op::GetAttr<nnvm::compiler::FTVMWeightPrepack>("FTVMWeightPrepack");

  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = src.GetAttr<DTypeVector>("dtype");
  const IndexedGraph& idx_graph = src.indexed_graph();

  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;

    nnvm::compiler::FTVMWeightPrepack fn_prepack = fweight_prepack.get(n->op(), nullptr);
    if (fn_prepack == nullptr) return false;

    // construct parameters for registered function
    std::vector<const Symbol*> op_inputs;
    tvm::Array<tvm::Tensor> tensor_infos;
    CHECK_EQ(n->num_inputs(), idx_graph[nid].inputs.size());
    for (uint32_t i = 0; i < n->num_inputs(); ++i) {
      const nnvm::NodeEntry& input = n->inputs[i];
      // input operator
      Symbol* op_input = new Symbol();
      op_input->outputs.push_back(input);
      op_inputs.push_back(static_cast<const Symbol*>(op_input));

      // input tinfo, extract from the original graph
      // because it was where infer_shape & infer_type applied.
      tvm::Array<tvm::Tensor> op_output_tinfos =
        GetTensorInfo(idx_graph, idx_graph[nid].inputs[i].node_id, shape_vec, dtype_vec);
      tensor_infos.push_back(op_output_tinfos[input.index]);
    }
    // callback registered function to get a new operator.
    auto op = fn_prepack(n->attrs, op_inputs, tensor_infos);
    std::for_each(op_inputs.begin(), op_inputs.end(), [](const Symbol* s){ delete s; });
    *ret = op.outputs;
    return true;
  };

  return nnvm::compiler::GraphTransform(src, transform);
}

// register pass
NNVM_REGISTER_PASS(PrePack)
.describe("Return a pre-packed graph of src")
.set_body(PrePack)
.set_change_graph(true);

}  // namespace
}  // namespace pass
}  // namespace nnvm
