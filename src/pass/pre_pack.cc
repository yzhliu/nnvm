#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/pass_functions.h>
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

inline Symbol GetSymbol(NodeEntry &output) {
  Symbol sym;
  sym.outputs.push_back(output);
  return sym;
}

Graph PrePack(const Graph& src) {
  static auto& fweight_prepack =
    Op::GetAttr<nnvm::compiler::FTVMWeightPrepack>("FTVMWeightPrepack");

  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = src.GetAttr<DTypeVector>("dtype");
  const IndexedGraph& idx_graph = src.indexed_graph();

  /*
  auto &shape = shape_vec[0];
  fprintf(stderr, "Shape[0] ndim: %d\n", shape.ndim());
  fprintf(stderr, "Shape[1] ndim: %d\n", shape_vec[1].ndim());
   */

  std::unordered_map<const Node*, uint32_t> node_indexes;
  for (uint32_t nid = 0; nid < idx_graph.num_nodes(); ++nid) {
    const auto &inode = idx_graph[nid];
    node_indexes[inode.source] = nid;
  }

  std::unordered_map<const Node*, tvm::Array<tvm::Tensor> > tinfos;
  std::unordered_map<const Node*, Symbol> replace_symbol;

  DFSVisit(src.outputs, [&replace_symbol, &tinfos, &idx_graph, &node_indexes, &shape_vec, &dtype_vec](const NodePtr& n) {
    std::string op_name = n->is_variable() ? "Variable" : n->op()->name;

    // record the shape of each node
    if (tinfos.count(n.get()) == 0) {
      tinfos[n.get()] = tvm::Array<tvm::Tensor>{};
    }

    CHECK(node_indexes.count(n.get())) << "Missing node in IndexedGraph.";
    uint32_t nid = node_indexes[n.get()];
    for (uint32_t i = 0; i < n->num_outputs(); ++i) {
      tvm::Array<tvm::Expr> shape;
      for (int64_t x : shape_vec[idx_graph.entry_id(nid, i)]) {
        CHECK_LE(x, static_cast<int64_t>(std::numeric_limits<int>::max()));
        shape.push_back(tvm::make_const(tvm::Int(32), x));
      }

      auto it = tinfos.find(n.get());
      CHECK(it != tinfos.end());
      tvm::Array<tvm::Tensor>& vec = it->second;
      vec.push_back(tvm::placeholder(shape, GetTVMType(dtype_vec[idx_graph.entry_id(nid, i)])));
    }

    if (!n->is_variable()) {
      // compose for replaced symbol
      for (uint32_t i = 0; i < n->num_inputs(); ++i) {
        auto& input = n->inputs[i];
        if (replace_symbol.count(input.node.get())) {
          auto& replaced_sym = replace_symbol[input.node.get()];
          CHECK_EQ(replaced_sym.outputs.size(), 1) << "Pre-pack only support operators have one output.";
          n->inputs[i] = replaced_sym.outputs[0];
        }
      }

      nnvm::compiler::FTVMWeightPrepack fn_prepack = fweight_prepack.get(n->op(), nullptr);
      if (fn_prepack != nullptr) {
        fprintf(stderr, "Get FTVMWeightPrepack!\n");
        CHECK_EQ(n->num_outputs(), 1) << "Pre-pack only support operators have one output.";

        std::vector<const Symbol*> input_syms;
        std::unordered_set<const Node*> visited;
        tvm::Array<tvm::Tensor> input_shapes;

        for (auto &input : n->inputs) {
          auto *input_sym = new Symbol();
          input_sym->outputs.push_back(input);
          input_syms.push_back(static_cast<const Symbol*>(input_sym));

          if (visited.count(input.node.get())) continue;
          visited.insert(input.node.get());
          auto &placeholders = tinfos[input.node.get()];
          for (auto &ph : placeholders) {
            input_shapes.push_back(ph);
          }
        }

        auto s = fn_prepack(n->attrs, input_syms, input_shapes);
        replace_symbol[n.get()] = s;

        for (auto iter = input_syms.begin(); iter != input_syms.end(); ++iter) {
          delete *iter;
        }
      }
    }
    uint32_t idx = node_indexes.count(n.get()) ? node_indexes[n.get()] : 100000;
    fprintf(stderr, "Visit node [%s](index = %d) with %d inputs, %d outputs, %d deps.\n",
            op_name.c_str(), idx,
            n->num_inputs(), n->num_outputs(), n->control_deps.size());
  });
  return src;
}

// register pass
NNVM_REGISTER_PASS(PrePack)
.describe("Return a pre-packed graph of src")
.set_body(PrePack)
.set_change_graph(true);

}  // namespace
}  // namespace pass
}  // namespace nnvm

