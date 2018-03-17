/*!
 *  Copyright (c) 2017 by Contributors
 * \file convolution.cc
 * \brief Convolution operators
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <tvm/tensor.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/compiler/op_attr_types.h>
#include <tvm/tvm.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn.h"


using tvm::Tensor;
using tvm::Array;
using nnvm::compiler::FTVMCompute;
using nnvm::compiler::FTVMLayoutRequest;

namespace nnvm {
namespace top {

// conv2d
DMLC_REGISTER_PARAMETER(Conv2DParam);

inline bool Conv2DInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  dshape = ConvertLayout(dshape, param.layout, kNCHW);

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

  TShape wshape({param.channels / param.groups,
                 dshape[1] / param.groups,
                 param.kernel_size[0],
                 param.kernel_size[1]});

  wshape = ConvertLayout(wshape, kNCHW, param.layout);
  wshape[0] *= param.groups;

  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kWeight, wshape);
  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape,
                            Conv2DParam::kBias, TShape({param.channels}));
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  TShape oshape({dshape[0], param.channels, 0, 0});
  if (dshape[2] != 0) {
    oshape[2] = (dshape[2] + param.padding[0] * 2 - dilated_ksize_y) / param.strides[0] + 1;
  }
  if (dshape[3] != 0) {
    oshape[3] = (dshape[3] + param.padding[1] * 2 - dilated_ksize_x) / param.strides[1] + 1;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0,
                           ConvertLayout(oshape, kNCHW, param.layout));
  // Perform incomplete shape inference. Fill in the missing values in data shape.
  // 1) We can always fill in the batch_size.
  // 2) We can back-calculate the input height/width if the corresponding stride is 1.
  oshape = ConvertLayout((*out_shape)[0], param.layout, kNCHW);
  dshape[0] = oshape[0];
  if (oshape[2] && param.strides[0] == 1) {
    dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param.padding[0];
  }
  if (oshape[3] && param.strides[1] == 1) {
    dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param.padding[1];
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kData,
                          ConvertLayout(dshape, kNCHW, param.layout));
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

NNVM_REGISTER_OP(conv2d)
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DParam>)
.set_attr<FInferShape>("FInferShape", Conv2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FTVMLayoutRequest>(
  "FTVMLayoutRequest", [](const NodeAttrs& attrs,
                          std::vector<TLayoutInfo> *ilayouts,
                          std::vector<TLayoutInfo> *olayouts) {
  const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
  const TLayoutInfo& out_layout = LayoutFlagStr(param.layout);
  if (param.use_bias) {
    CHECK_EQ(ilayouts->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(ilayouts->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(olayouts->size(), 1U);
  olayouts->at(0) = out_layout;
  for (uint32_t i = 0; i < ilayouts->size(); ++i) {
    if (ilayouts->at(i) != "__undef__" &&
        !CheckLayoutConvertible(ilayouts->at(i), out_layout)) {
      return false;
    }
    ilayouts->at(i) = out_layout;
  }
  return true;
})
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DParam>)
.set_support_level(2)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return MakeGradNode("_conv2d_grad", n,
                        {ograds[0], n->inputs[Conv2DParam::kData],
                         n->inputs[Conv2DParam::kWeight]},
                        n->attrs.dict);
});


struct Conv2DNCHWcParam : public dmlc::Parameter<Conv2DNCHWcParam> {
  int channels;
  TShape kernel_size;
  TShape strides;
  TShape padding;
  TShape dilation;
  int groups;
  bool use_bias;
  int ic_bn;
  int oc_bn;

  DMLC_DECLARE_PARAMETER(Conv2DNCHWcParam) {
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
    DMLC_DECLARE_FIELD(ic_bn).set_default(16)
    .describe("Input channel block size.");
    DMLC_DECLARE_FIELD(oc_bn).set_default(16)
    .describe("Output channel block size.");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};
DMLC_REGISTER_PARAMETER(Conv2DNCHWcParam);

inline bool Conv2DNoPackInferShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_shape,
                                   std::vector<TShape>* out_shape) {
  const Conv2DNCHWcParam& param = nnvm::get<Conv2DNCHWcParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;

  CHECK_EQ(dshape.ndim(), 5U) << "Input data should be 5D";
  CHECK_EQ(param.kernel_size.ndim(), 2U);
  CHECK_EQ(param.strides.ndim(), 2U)
    << "incorrect stride size: " << param.strides;
  CHECK_EQ(param.dilation.ndim(), 2U)
    << "incorrect dilate size: " << param.dilation;
  CHECK_EQ(dshape[1] % param.groups, 0U)
    << "input channels must divide group size";
  CHECK_EQ(param.channels % param.groups, 0U)
    << "output channels must divide group size";

//  CHECK_EQ(dshape[4], param.ic_bn)
//    << "input channels block must == ic_bn";
  CHECK_EQ(param.channels % param.oc_bn, 0U)
    << "output channels must divide oc_bn";

  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape,
                            Conv2DNCHWcParam::kBias, TShape({param.channels/param.oc_bn, param.oc_bn}));
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  // oshape = [n, C, h, w, c]
  TShape oshape({dshape[0], param.channels/param.oc_bn, 0, 0, param.oc_bn});
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
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DNCHWcParam::kData, dshape);
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

NNVM_REGISTER_OP(conv2d_nChwc)
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).
)code" NNVM_ADD_FILELINE)
.add_argument("data", "5D Tensor", "Packed input data.")
.add_argument("weight", "6D Tensor", "Packed weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DNCHWcParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DNCHWcParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DNCHWcParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DNCHWcParam>)
.set_attr<FInferShape>("FInferShape", Conv2DNoPackInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FTVMLayoutRequest>(
"FTVMLayoutRequest", [](const NodeAttrs& attrs,
                        std::vector<TLayoutInfo> *ilayouts,
                        std::vector<TLayoutInfo> *olayouts) {
  const Conv2DNCHWcParam& param = nnvm::get<Conv2DNCHWcParam>(attrs.parsed);
  TLayoutInfo in_layout;
  TLayoutInfo out_layout;
  switch (param.ic_bn) {
    case 3:
      in_layout = "NCHW3c";
      break;
    case 8:
      in_layout = "NCHW8c";
      break;
    case 16:
    default:
      in_layout = "NCHW16c";
      break;
  }
  switch (param.oc_bn) {
    case 3:
      out_layout = "NCHW3c";
      break;
    case 8:
      out_layout = "NCHW8c";
      break;
    case 16:
    default:
      out_layout = "NCHW16c";
      break;
  }

  if (param.use_bias) {
    CHECK_EQ(ilayouts->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(ilayouts->size(), 2U) << "Input:[data, weight]";
  }
  // TODO: decide arg layout. now we assume arg layout has been correctly converted.
  ilayouts->at(0) = in_layout;

  CHECK_EQ(olayouts->size(), 1U);
  olayouts->at(0) = out_layout;

  return true;
})
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DNCHWcParam>)
.set_support_level(2);


NNVM_REGISTER_OP(_conv2d_grad)
  .describe(R"code(2D convolution grad.

)code" NNVM_ADD_FILELINE)
.add_argument("ograd", "4D Tensor", "Output grad.")
.add_argument("data", "4D Tensor", "Input data of conv2d.")
.add_argument("weight", "4D Tensor", "Input weight.")
.set_num_inputs(3)
.set_num_outputs(UseBiasNumInputs<Conv2DParam>)
.set_attr<FListOutputNames>("FListOutputNames", UseBiasListInputNames<Conv2DParam>)
.set_attr_parser(ParamParser<Conv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DParam>)
.set_attr<FInferShape>(
  "FInferShape", [](const nnvm::NodeAttrs& attrs,
                    std::vector<TShape>* in_attrs,
                    std::vector<TShape>* out_attrs) {
    const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, Conv2DParam::kData, in_attrs->at(1));
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, Conv2DParam::kWeight, in_attrs->at(2));
    if (param.use_bias) {
      NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, Conv2DParam::kBias, TShape({param.channels}));
    }
    return true;
})
.set_attr<FInferType>("FInferType", ElemwiseType<3, -1>)
.set_attr<TIsBackward>("TIsBackward", true);


DMLC_REGISTER_PARAMETER(Conv2DTransposeParam);

inline bool Conv2DTransposeInferShape(const nnvm::NodeAttrs& attrs,
                                      std::vector<TShape>* in_shape,
                                      std::vector<TShape>* out_shape) {
  const Conv2DTransposeParam& param = nnvm::get<Conv2DTransposeParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  const TShape& dshape = (*in_shape)[Conv2DTransposeParam::kData];
  if (dshape.ndim() ==  0) return false;
  TShape dshape_nchw = ConvertLayout(dshape, param.layout, kNCHW);

  CHECK_EQ(dshape_nchw[1] % param.groups, 0U)
      << "input num_filter must divide group size";
  CHECK_EQ(param.channels % param.groups, 0U)
      << "output num_filter must divide group size";
  CHECK_EQ(param.kernel_size.ndim(), 2U)
      << "incorrect kernel size: " << param.kernel_size;
  CHECK_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;
  CHECK_EQ(param.dilation.ndim(), 2U)
      << "incorrect dilate size: " << param.dilation;

  TShape wshape({dshape_nchw[1],
                 param.channels / param.groups,
                 param.kernel_size[0],
                 param.kernel_size[1]});
  wshape = ConvertLayout(wshape, kNCHW, param.layout);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DTransposeParam::kWeight, wshape);

  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape,
                            Conv2DTransposeParam::kBias,
                            TShape({param.channels}));
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  // output shape.
  TShape oshape({dshape_nchw[0], param.channels, 0, 0});
  oshape[2] = (param.strides[0] * (dshape_nchw[2] - 1) + dilated_ksize_y -
               2 * param.padding[0] + param.output_padding[0]);

  oshape[3] = (param.strides[1] * (dshape_nchw[3] - 1) + dilated_ksize_x -
               2 * param.padding[1] + param.output_padding[1]);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0,
                           ConvertLayout(oshape, kNCHW, param.layout));
  return true;
}

NNVM_REGISTER_OP(conv2d_transpose)
.describe(R"code(Transposed 2D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (in_channels, channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

            out_height and out_width are calculated as::
                out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
                out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DTransposeParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DTransposeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DTransposeParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DTransposeParam>)
.set_attr<FInferShape>("FInferShape", Conv2DTransposeInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DTransposeParam>)
.set_support_level(2);

}  // namespace top
}  // namespace nnvm
