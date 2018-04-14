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
inline TShape ConvertLayout(TShape src, const Layout& src_layout, const Layout& dst_layout) {
  if (src_layout == dst_layout) return src;
  else if (!src_layout.IsDefined()) {
    LOG(FATAL) << "cannot convert undefined layout to " << dst_layout;
  } else if (!dst_layout.IsDefined()) {
    LOG(FATAL) << "cannot convert " << src_layout << " to undefined layout";
  }

  CHECK(src_layout.Convertible(dst_layout)) << "cannot convert from "
                                            << src_layout << " to " << dst_layout;

  TShape dst(dst_layout.ndim());
  for (size_t i = 0; i < src_layout.ndim(); ++i) {
    Layout::LayoutAxis src_axis = src_layout[i];
    if (Layout::IsMajorAxis(src_axis)) {
      int dst_major_pos = dst_layout.PosMajor(src_axis);
      int dst_minor_pos = dst_layout.PosMinor(src_axis);
      int src_minor_pos = src_layout.PosMinor(src_axis);
      int src_factor = src_layout.FactorSize(src_axis);
      int dst_factor = dst_layout.FactorSize(src_axis);

      uint32_t src_axis_size = src[i];
      if (src_minor_pos >= 0) {
        if (src_factor == -1) src_factor = src[src_minor_pos];
        CHECK_EQ(src_factor, src[src_minor_pos]) << "src shape " << src
                                                 << " does not agree with layout " << src_layout;
        src_axis_size *= src_factor;
      }

      dst[dst_major_pos] = src_axis_size;
      if (dst_minor_pos >= 0) {
        CHECK_GT(dst_factor, 0);
        CHECK_LE(dst_factor, src_axis_size) << "Converting " << src
                                            << " from " << src_layout
                                            << " to " << dst_factor
                                            << ": cannot split axis size of "
                                            << src_axis_size << " by " << dst_factor;
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
