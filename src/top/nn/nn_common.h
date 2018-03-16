/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn_common.h
 * \brief Common utilities for nn ops.
 */
#ifndef NNVM_TOP_NN_NN_COMMON_H_
#define NNVM_TOP_NN_NN_COMMON_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
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

  TShape dst;
  if (src.ndim() == 3) {
    dst = src;
    switch (src_layout) {
      case kNCW: break;
      case kNWC: {
        std::swap(dst[1], dst[2]);
        break;
      }
      default: {
        LOG(FATAL) << "invalid layout for 3d shape " << LayoutFlagStr(src_layout);
      }
    }
    switch (dst_layout) {
      case kNCW: break;
      case kNWC: {
        std::swap(dst[1], dst[2]);
        break;
      }
      default: {
        LOG(FATAL) << "invalid layout for 3d shape " << LayoutFlagStr(dst_layout);
      }
    }
  } else if (src.ndim() == 4) {
    int ndim = (dst_layout == kNCHW3c || dst_layout == kNCHW16c || dst_layout == kNCHW8c) ? 5 : 4;
    dst = TShape(ndim);
    switch (src_layout) {
      case kNCHW: {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        break;
      }
      case kNHWC: {
        dst[0] = src[0];
        dst[2] = src[1];
        dst[3] = src[2];
        dst[1] = src[3];
        break;
      }
      default: {
        LOG(FATAL) << "invalid layout for 4d shape " << LayoutFlagStr(src_layout);
      }
    }
    src = dst;
    switch (dst_layout) {
      case kNCHW: break;
      case kNHWC: {
        dst[1] = src[2];
        dst[2] = src[3];
        dst[3] = src[1];
        break;
      }
      case kNCHW3c: {
        dst[1] = dst[1] / 3;
        dst[4] = 3;
        break;
      }
      case kNCHW8c: {
        dst[1] = dst[1] / 8;
        dst[4] = 8;
        break;
      }
      case kNCHW16c: {
        dst[1] = dst[1] / 16;
        dst[4] = 16;
        break;
      }
      default: {
        LOG(FATAL) << "invalid layout for 4d shape " << LayoutFlagStr(dst_layout);
      }
    }
  } else if (src.ndim() == 5) {
    if (src_layout == kNCHW16c || src_layout == kNCHW8c) {
      int ndim = (dst_layout == kNCHW16c || dst_layout == kNCHW8c) ? 5 : 4;
      dst = TShape(ndim);
      dst[0] = src[0];
      dst[1] = src[1] * src[4];
      dst[2] = src[2];
      dst[3] = src[3];
      switch (dst_layout) {
        case kNCHW: break;
        case kNHWC: {
          std::swap(dst[1], dst[3]);
          std::swap(dst[1], dst[2]);
          break;
        }
        case kNCHW3c: {
          dst[1] = dst[1] / 3;
          dst[4] = 3;
          break;
        }
        case kNCHW8c: {
          dst[1] = dst[1] / 8;
          dst[4] = 8;
          break;
        }
        case kNCHW16c: {
          dst[1] = dst[1] / 16;
          dst[4] = 16;
          break;
        }
        default: {
          LOG(FATAL) << "invalid layout for 5d shape " << LayoutFlagStr(dst_layout);
        }
      }
    } else {
      dst = src;
      switch (src_layout) {
        case kNCDHW: break;
        case kNDHWC: {
          dst[2] = src[1];
          dst[3] = src[2];
          dst[4] = src[3];
          dst[1] = src[4];
          break;
        }
        default: {
          LOG(FATAL) << "invalid layout for 5d shape " << LayoutFlagStr(src_layout);
        }
      }
      src = dst;
      switch (dst_layout) {
        case kNCDHW: break;
        case kNDHWC: {
          dst[1] = src[2];
          dst[2] = src[3];
          dst[3] = src[4];
          dst[4] = src[1];
          break;
        }
        default: {
          LOG(FATAL) << "invalid layout for 5d shape " << LayoutFlagStr(dst_layout);
        }
      }
    }
  } else {
    LOG(FATAL) << "no layout option for " << dst.ndim() << " dimensions";
  }
  return dst;
}

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_NN_NN_COMMON_H_
