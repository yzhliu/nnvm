/*!
 *  Copyright (c) 2018 by Contributors
 * \file memory_reorder.cc
 * \brief Property def of memory reorder operators.
 */
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <topi/transform.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn/dense.h"
#include "topi/nn.h"
#include "topi/nn/softmax.h"

namespace nnvm {
namespace top {

using tvm::Tensor;
using tvm::Array;
using nnvm::compiler::FTVMCompute;

struct ReorderParam : public dmlc::Parameter<ReorderParam> {
  int oc_bn;
  int ic_bn;

  DMLC_DECLARE_PARAMETER(ReorderParam) {
    DMLC_DECLARE_FIELD(oc_bn).set_lower_bound(1)
    .describe("Output channel number of block.");
    DMLC_DECLARE_FIELD(ic_bn).set_lower_bound(1)
    .describe("Input channel number of block.");
  }
};

DMLC_REGISTER_PARAMETER(ReorderParam);

inline bool ReorderInferShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const ReorderParam& param = nnvm::get<ReorderParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  const TShape& shp = (*in_shape)[0];
  if (shp.ndim() == 0) return false;

  TShape ret(shp.ndim() + 2);
  auto h = shp[2];
  auto w = shp[3];
  if (h == 1 && w == 1) {
    // (oc, ic, h, w) -> (OC, IC, ic, oc, h, w)
    ret[0] = shp[0] / param.oc_bn;
    ret[1] = shp[1] / param.ic_bn;
    ret[2] = param.ic_bn;
    ret[3] = param.oc_bn;
    ret[4] = h;
    ret[5] = w;
  } else {
    // (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
    ret[0] = shp[0] / param.oc_bn;
    ret[1] = shp[1] / param.ic_bn;
    ret[2] = shp[2];
    ret[3] = shp[3];
    ret[4] = param.ic_bn;
    ret[5] = param.oc_bn;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, ret);
  return true;
}

NNVM_REGISTER_OP(reorder)
.describe(R"code(Applies a memory reorder
)code" NNVM_ADD_FILELINE)
.add_argument("data", "nD Tensor", "Input data.")
.add_arguments(ReorderParam::__FIELDS__())
.set_attr_parser(ParamParser<ReorderParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReorderParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", ReorderInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FTVMCompute>(
"FTVMCompute", [](const NodeAttrs& attrs,
                  const Array<Tensor>& inputs,
                  const Array<Tensor>& out_info) {
  const ReorderParam& param = nnvm::get<ReorderParam>(attrs.parsed);
  return Array<Tensor>{ topi::reorder(inputs[0], out_info[0]->shape, param.oc_bn, param.ic_bn) };
})
.set_support_level(1);

}  // namespace top
}  // namespace nnvm
