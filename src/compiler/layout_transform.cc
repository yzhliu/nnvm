/*!
 *  Copyright (c) 2017 by Contributors
 * \file layout_transform.cc
 * \brief Transforms layout.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/layout.h>

namespace nnvm {
namespace compiler {

nnvm::NodePtr CreateLayoutTransformNode(const Layout& src,
                                        const Layout& dst) {
  static const nnvm::Op* trans_op = nnvm::Op::Get("__layout_transform__");
  static int count = 0;
  nnvm::NodePtr n = nnvm::Node::Create();
  n->attrs.op = trans_op;
  n->attrs.name = src.name() + "_to_" + dst.name() + std::to_string(count++);
  n->attrs.dict["src_layout"] = src.name();
  n->attrs.dict["dst_layout"] = dst.name();
  n->op()->attr_parser(&(n->attrs));
  return n;
}

using LayoutAttrDict = std::unordered_map<const Node*, std::vector<Layout> >;

/*!
 * \brief A simple layout transform pass that will
 *  insert layout transform nodes automatically.
 */
nnvm::Graph LayoutTransform(nnvm::Graph src) {
  static auto& op_infer_layout =
    nnvm::Op::GetAttr<FInferLayout>("FInferLayout");

  const std::vector<Layout>& input_layouts =
      src.GetAttr<std::vector<Layout> >("layout_inputs");

  const IndexedGraph& idx = src.indexed_graph();
  std::vector<nnvm::NodePtr> mirror_vec(idx.num_nodes(), nullptr);

  // (new) NodePtr -> output_layouts
  LayoutAttrDict new_layouts;

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    nnvm::NodePtr new_node = nnvm::Node::Create();
    *new_node = *(inode.source);
    if (new_node->is_variable()) {
      // Variable node. No operator. Only one output entry.
      auto input_iter = std::find(
        idx.input_nodes().cbegin(), idx.input_nodes().cend(), nid);
      CHECK(input_iter != idx.input_nodes().cend());
      long input_id = std::distance(idx.input_nodes().cbegin(), input_iter);
      new_layouts[new_node.get()] = { input_layouts[input_id] };
      mirror_vec[nid] = new_node;
      continue;
    }

    // set up output and input layouts
    std::vector<Layout> request_ilayouts(new_node->num_inputs(), Layout::Undef());
    for (size_t i = 0; i < new_node->num_inputs(); ++i) {
      const IndexedGraph::NodeEntry& input_entry = inode.inputs[i];
      const NodePtr& new_input_node = mirror_vec[input_entry.node_id];
      CHECK(new_input_node != nullptr);

      // fill inputs by previous node (DFS order) inferred layouts.
      const auto& layouts_iter = new_layouts.find(new_input_node.get());
      CHECK(layouts_iter != new_layouts.end());
      request_ilayouts[i] = layouts_iter->second[input_entry.index];
    }
    // layouts produced by previous node.
    std::vector<Layout> produce_ilayouts(request_ilayouts);
    // input layouts from last pass of LayoutTransform (if apply)
    std::vector<Layout> last_request_ilayouts(new_node->num_inputs(), Layout::Undef());
    // fill outputs by last pass of LayoutTransform (if apply)
    std::vector<Layout> produce_olayouts(new_node->num_outputs(), Layout::Undef());
    if (src.HasAttr("layout")) {
      const auto& layouts = src.GetAttr<std::vector<Layout> >("layout");
      for (uint32_t i = 0; i < new_node->num_outputs(); ++i) {
        produce_olayouts[i] = layouts[idx.entry_id(nid, i)];
      }
      for (uint32_t i = 0; i < new_node->num_inputs(); ++i) {
        last_request_ilayouts[i] = layouts[idx.entry_id(inode.inputs[i])];
      }
    }

    const auto& flayout = op_infer_layout[new_node->op()];
    CHECK(flayout != nullptr) << "Attribute FInferLayout"
                              << " is not registered by op " << inode.source->op()->name
                              << " we are not able to complete layout transform.";
    CHECK(flayout(new_node->attrs, &request_ilayouts, &last_request_ilayouts, &produce_olayouts))
        << "Layout infer fail";
    CHECK_EQ(request_ilayouts.size(), new_node->num_inputs());
    CHECK_EQ(produce_olayouts.size(), new_node->num_outputs());
    for (const Layout& l : request_ilayouts) {
      CHECK(!l.IsDefined() || l.IsAxisFactorComplete())
        << inode.source->op()->name << " generates incomplete input layout " << l;
    }
    for (const Layout& l : produce_olayouts) {
      CHECK(!l.IsDefined() || l.IsAxisFactorComplete())
        << inode.source->op()->name << " generates incomplete output layout " << l;
    }

    // update new layouts
    new_layouts[new_node.get()] = std::move(produce_olayouts);

    for (uint32_t i = 0; i < inode.inputs.size(); ++i) {
      const auto& e = inode.inputs[i];
      const nnvm::NodePtr& in = mirror_vec[e.node_id];
      new_node->inputs[i] = nnvm::NodeEntry{in, e.index, e.version};

      // insert layout_transform if necessary
      const Layout& produce = produce_ilayouts[i];
      const Layout& request = request_ilayouts[i];
      if (produce != request && produce.IsDefined()) {
        nnvm::NodePtr tnode = CreateLayoutTransformNode(produce, request);
        tnode->attrs.name = idx[e.node_id].source->attrs.name + "_" + request.name();
        tnode->inputs.emplace_back(new_node->inputs[i]);
        nnvm::NodeEntry tnode_output{tnode, 0, 0};
        new_node->inputs[i] = tnode_output;
        // layout produced by LayoutTransformNode
        new_layouts[tnode.get()] = {request};
      }
    }
    mirror_vec[nid] = new_node;
  }

  std::vector<nnvm::NodeEntry> outputs;
  for (const auto& e : idx.outputs()) {
    outputs.emplace_back(nnvm::NodeEntry{mirror_vec[e.node_id], e.index, e.version});
  }

  nnvm::Graph ret;
  ret.outputs = outputs;
  // restore the layouts to return graph
  const auto& ret_idx = ret.indexed_graph();
  std::vector<Layout> ret_layouts(ret_idx.num_node_entries(), Layout::Undef());
  for (uint32_t nid = 0; nid < ret_idx.num_nodes(); ++nid) {
    const auto& inode = ret_idx[nid];
    const auto& layout_iter = new_layouts.find(inode.source);
    if (layout_iter != new_layouts.end()) {
      for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
        ret_layouts[ret_idx.entry_id(nid, i)] = std::move(layout_iter->second[i]);
      }
    }
  }

  // cannot call indexed_graph() before return the origin Graph,
  // thus create a new one
  nnvm::Graph new_ret;
  new_ret.outputs = std::move(outputs);
  new_ret.attrs["layout"] = std::make_shared<any>(std::move(ret_layouts));

  return new_ret;
}

// register pass
NNVM_REGISTER_PASS(LayoutTransform)
.describe("Return a layout-transformed graph of src.")
.set_body(LayoutTransform)
.provide_graph_attr("layout")
.set_change_graph(true);

DMLC_JSON_ENABLE_ANY(LayoutVector, list_layout);

}  // namespace compiler
}  // namespace nnvm
