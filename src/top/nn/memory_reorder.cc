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
  bool kernel_1x1;

  DMLC_DECLARE_PARAMETER(ReorderParam) {
    DMLC_DECLARE_FIELD(oc_bn).set_lower_bound(1)
    .describe("Output channel number of block.");
    DMLC_DECLARE_FIELD(ic_bn).set_lower_bound(1)
    .describe("Input channel number of block.");
    DMLC_DECLARE_FIELD(kernel_1x1).set_default(false)
    .describe("Whether it is 1x1 kernel.");
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
  if (param.kernel_1x1) {
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
    ret[2] = h;
    ret[3] = w;
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
  return Array<Tensor>{ topi::reorder(inputs[0], out_info[0]->shape,
                                      param.oc_bn, param.ic_bn, param.kernel_1x1) };
})
.set_support_level(1);

// reorder for bn input mean & var
struct DataReorderParam : public dmlc::Parameter<DataReorderParam> {
  int bn;
  DMLC_DECLARE_PARAMETER(DataReorderParam) {
    DMLC_DECLARE_FIELD(bn).set_lower_bound(1)
    .describe("Channel number of block.");
  }
};
DMLC_REGISTER_PARAMETER(DataReorderParam);

inline bool BNReorderInferShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape>* in_shape,
                                std::vector<TShape>* out_shape) {
  // c -> Cc
  fprintf(stderr, "start BNReorderInferShape\n");
  const DataReorderParam& param = nnvm::get<DataReorderParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  const TShape& shp = (*in_shape)[0];
  CHECK_EQ(shp.ndim(), 1U);

  TShape ret(2);
  ret[0] = shp[0] / param.bn;
  ret[1] = param.bn;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, ret);
  fprintf(stderr, "end BNReorderInferShape\n");
  return true;
}

NNVM_REGISTER_OP(bn_reorder)
.describe(R"code(Applies a memory reorder for batch norm mean & var
)code" NNVM_ADD_FILELINE)
.add_argument("data", "1D Tensor", "Input data.")
.add_arguments(DataReorderParam::__FIELDS__())
.set_attr_parser(ParamParser<DataReorderParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DataReorderParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", BNReorderInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FTVMCompute>(
"FTVMCompute", [](const NodeAttrs& attrs,
                  const Array<Tensor>& inputs,
                  const Array<Tensor>& out_info) {
  const DataReorderParam& param = nnvm::get<DataReorderParam>(attrs.parsed);
  return Array<Tensor>{ topi::bnreorder(inputs[0], out_info[0]->shape, param.bn) };
})
.set_support_level(1);

// reorder for conv input data
inline bool DataReorderBackInferShape(const nnvm::NodeAttrs& attrs,
                                      std::vector<TShape>* in_shape,
                                      std::vector<TShape>* out_shape) {
  // nChwc -> nchw
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  const TShape& shp = (*in_shape)[0];
  if (shp.ndim() == 0) return false;

  TShape ret(shp.ndim() - 1);
  auto n = shp[0];
  auto C = shp[1];
  auto h = shp[2];
  auto w = shp[3];
  auto c = shp[4];

  ret[0] = n;
  ret[1] = C * c;
  ret[2] = h;
  ret[3] = w;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, ret);
  return true;
}

NNVM_REGISTER_OP(data_reorder_back)
.describe(R"code(Applies a memory reorder back for conv input data
)code" NNVM_ADD_FILELINE)
.add_argument("data", "1D Tensor", "Input data.")
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", DataReorderBackInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FTVMCompute>(
"FTVMCompute", [](const NodeAttrs& attrs,
                  const Array<Tensor>& inputs,
                  const Array<Tensor>& out_info) {
  return Array<Tensor>{ topi::data_reorder_back(inputs[0], out_info[0]->shape) };
})
.set_support_level(1);

}  // namespace top
}  // namespace nnvm
