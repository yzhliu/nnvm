/*!
 *  Copyright (c) 2018 by Contributors
 * \file batch_norm_inference.cc
 * \brief Property def of batch_norm (inference) operators.
 */
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <nnvm/top/contrib/batch_norm_inference.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn/dense.h"
#include "topi/nn.h"

using namespace nnvm::top;

namespace nnvm {
namespace top {
namespace contrib {

DMLC_REGISTER_PARAMETER(BatchNormInferenceParam);

inline bool BatchNormInferNCHWcShape(const nnvm::NodeAttrs& attrs,
                                     std::vector<TShape>* in_shape,
                                     std::vector<TShape>* out_shape) {
  CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, moving_mean, moving_var]";
  CHECK_EQ(out_shape->size(), 3U);
  const TShape &dshape = in_shape->at(0);
  CHECK_EQ(dshape.ndim(), 5U) << "Input data must be 5-D.";

  int channel_chunk = dshape[1];
  int channel_block = dshape[4];
  TShape bshape({channel_chunk * channel_block});
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BatchNormInferenceParam::kGamma, bshape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BatchNormInferenceParam::kBeta, bshape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BatchNormInferenceParam::kMovingMean, bshape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BatchNormInferenceParam::kMovingVariance, bshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 1, bshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 2, bshape);
  return true;
}

NNVM_REGISTER_OP(_contrib_batch_norm_inference_nChwc)
.describe(R"(Batch normalization inference layer for nChwc layout (Ioffe and Szegedy, 2014).
Normalizes the input at each batch, i.e. applies a transformation
that maintains the mean activation close to 0 and the activation
standard deviation close to 1.

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta`` have shape *(k,)*.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
axis to be the last item in the input shape.

.. note::
    This operator can be optimized away for inference.
)" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input to which dropout will be applied")
.add_argument("gamma", "Tensor", "The gamma scale factor")
.add_argument("beta", "Tensor", "The beta offset factor")
.add_argument("moving_mean", "Tensor", "running mean of input")
.add_argument("moving_var", "Tensor", "running variance of input")
.add_arguments(BatchNormInferenceParam::__FIELDS__())
.set_attr_parser(ParamParser<BatchNormInferenceParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<BatchNormInferenceParam>)
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr<FInferShape>("FInferShape", BatchNormInferNCHWcShape)
.set_attr<FInferType>("FInferType", ElemwiseType<5, 3>)
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta", "moving_mean", "moving_var"};
})
.set_attr<FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "var"};
})
.set_support_level(1);

}  // namespace contrib
}  // namespace top
}  // namespace nnvm
