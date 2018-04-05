/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn_common.h
 * \brief Common utilities for nn ops.
 */
#ifndef NNVM_TOP_NN_NN_COMMON_H_
#define NNVM_TOP_NN_NN_COMMON_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <nnvm/layout.h>
#include <nnvm/top/nn.h>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

namespace nnvm {
namespace top {

template<typename ParamType>
inline uint32_t UseBiasNumInputs(const NodeAttrs& attrs) {
  const ParamType& param = get<ParamType>(attrs.parsed);
  return param.use_bias ? 3 : 2;
}

template<typename ParamType>
inline std::vector<std::string> UseBiasListInputNames(const NodeAttrs& attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  if (param.use_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

/*!
 * \brief Convert shape in src_layout to shape in dst_layout
 * \param src original shape
 * \param src_layout layout of original shape
 * \param dst_layout target layout
 * \return shape in target layout
 */
inline TShape ConvertLayout(TShape src, int src_layout, int dst_layout) {
  if (src_layout == dst_layout) return src;
  else if (src_layout == kUndef) {
    LOG(FATAL) << "cannot convert undefined layout to " << LayoutFlagStr(dst_layout);
  } else if (dst_layout == kUndef) {
    LOG(FATAL) << "cannot convert " << LayoutFlagStr(src_layout) << " to undefined layout";
  }

  Layout slayout(LayoutFlagStr(src_layout));
  Layout dlayout(LayoutFlagStr(dst_layout));

  CHECK(slayout.ConvertibleTo(dlayout)) << "cannot convert from " << slayout.name
                                        << " to " << dlayout.name;

  TShape dst(dlayout.ndim());
  for (size_t i = 0; i < slayout.ndim(); ++i) {
    Layout::LayoutAxis src_axis = slayout[i];
    if (Layout::IsMajorAxis(src_axis)) {
      int dst_major_pos = dlayout.PosMajor(src_axis);
      int dst_minor_pos = dlayout.PosMinor(src_axis);
      int src_minor_pos = slayout.PosMinor(src_axis);
      uint32_t src_factor = slayout.FactorSize(src_axis);
      uint32_t dst_factor = dlayout.FactorSize(src_axis);

      uint32_t src_axis_size = src[i];
      if (src_minor_pos >= 0) {
        CHECK_EQ(src_factor, src[src_minor_pos]) << "src shape " << src
                                                 << " does not agree with layout " << slayout.name;
        src_axis_size *= src_factor;
      }

      dst[dst_major_pos] = src_axis_size;
      if (dst_minor_pos >= 0) {
        CHECK(dst_factor > 0);
        dst[dst_major_pos] /= dst_factor;
        dst[dst_minor_pos] = dst_factor;
      }
    }
  }
  return dst;
}

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_NN_NN_COMMON_H_
