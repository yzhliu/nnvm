/*!
 *  Copyright (c) 2018 by Contributors
 * \file convolution.cc
 * \brief Property def of convolution (contrib) operators.
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

namespace nnvm {
namespace top {
namespace contrib {

template<typename ParamType>
inline uint32_t UseBiasNumInputs(const nnvm::NodeAttrs& attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  return param.use_bias ? 3 : 2;
}

template<typename ParamType>
inline std::vector<std::string> UseBiasListInputNames(const nnvm::NodeAttrs& attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  if (param.use_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

struct Conv2DNCHWKernelPackedParam : public dmlc::Parameter<Conv2DNCHWKernelPackedParam> {
  int channels;
  TShape kernel_size;
  TShape strides;
  TShape padding;
  TShape dilation;
  int groups;
  bool use_bias;

  DMLC_DECLARE_PARAMETER(Conv2DNCHWKernelPackedParam) {
  DMLC_DECLARE_FIELD(channels)
    .describe("The dimensionality of the output space"
              "i.e. the number of output channels in the convolution.");
  DMLC_DECLARE_FIELD(kernel_size)
    .describe("Specifies the dimensions of the convolution window.");
  DMLC_DECLARE_FIELD(strides).set_default(TShape({1, 1}))
  .describe("Specifies the strides of the convolution.");
  DMLC_DECLARE_FIELD(padding).set_default(TShape({0, 0}))
  .describe("If padding is non-zero, then the input is implicitly zero-padded"
  "on both sides for padding number of points");
  DMLC_DECLARE_FIELD(dilation).set_default(TShape({1, 1}))
  .describe("Specifies the dilation rate to use for dilated convolution.");
  DMLC_DECLARE_FIELD(groups).set_default(1)
  .describe("Controls the connections between inputs and outputs."
  "At groups=1, all inputs are convolved to all outputs."
  "At groups=2, the operation becomes equivalent to having two convolution"
  "layers side by side, each seeing half the input channels, and producing"
  "half the output channels, and both subsequently concatenated.");
  DMLC_DECLARE_FIELD(use_bias).set_default(true)
  .describe("Whether the layer uses a bias vector.");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};

DMLC_REGISTER_PARAMETER(Conv2DNCHWKernelPackedParam);

inline bool Conv2DNCHWKernelPrePackInferShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_shape,
                                   std::vector<TShape>* out_shape) {
    const Conv2DNCHWKernelPackedParam& param = nnvm::get<Conv2DNCHWKernelPackedParam>(attrs.parsed);
    if (param.use_bias) {
        CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
    } else {
        CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
    }
    CHECK_EQ(out_shape->size(), 1U);

    TShape dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;

    CHECK_EQ(dshape.ndim(), 4U) << "Input data should be 4D";
    CHECK_EQ(param.kernel_size.ndim(), 2U);
    CHECK_EQ(param.strides.ndim(), 2U)
            << "incorrect stride size: " << param.strides;
    CHECK_EQ(param.dilation.ndim(), 2U)
            << "incorrect dilate size: " << param.dilation;
    CHECK_EQ(dshape[1] % param.groups, 0U)
            << "input channels must divide group size";
    CHECK_EQ(param.channels % param.groups, 0U)
            << "output channels must divide group size";

    if (param.use_bias) {
        NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape,
                                Conv2DNCHWKernelPackedParam::kBias, TShape({param.channels}));
    }
    // dilation
    dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
    dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
    // oshape = [n, c, h, w]
    TShape oshape({dshape[0], param.channels, 0, 0});
    if (dshape[2] != 0) {
        oshape[2] = (dshape[2] + param.padding[0] * 2 - dilated_ksize_y) / param.strides[0] + 1;
    }
    if (dshape[3] != 0) {
        oshape[3] = (dshape[3] + param.padding[1] * 2 - dilated_ksize_x) / param.strides[1] + 1;
    }
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    dshape[0] = oshape[0];
    if (oshape[2] && param.strides[0] == 1) {
        dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param.padding[0];
    }
    if (oshape[3] && param.strides[1] == 1) {
        dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param.padding[1];
    }
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DNCHWKernelPackedParam::kData, dshape);
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
        CHECK_LE(dilated_ksize_y, dshape[2] + 2 * param.padding[0])
                << "kernel size exceed input";
    }
    if (dshape[3] != 0) {
        CHECK_LE(dilated_ksize_x, dshape[3] + 2 * param.padding[1])
                << "kernel size exceed input";
    }
    return true;
}

NNVM_REGISTER_OP(_contrib_conv2d_nchwc_kernel_packed)
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).
)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "6D Tensor", "Packed weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DNCHWKernelPackedParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DNCHWKernelPackedParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DNCHWKernelPackedParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DNCHWKernelPackedParam>)
.set_attr<FInferShape>("FInferShape", Conv2DNCHWKernelPrePackInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FTVMLayoutRequest>(
"FTVMLayoutRequest", [](const NodeAttrs& attrs,
        std::vector<TLayoutInfo> *ilayouts,
        std::vector<TLayoutInfo> *olayouts) {
// TODO: decide arg layout. now we assume arg layout has been correctly converted.
ilayouts->at(0) = "NCHW";
CHECK_EQ(olayouts->size(), 1U);
olayouts->at(0) = "NCHW";
return true;
})
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DNCHWKernelPackedParam>)
.set_support_level(2);

}  // namespace contrib
}  // namespace top
}  // namespace nnvm
