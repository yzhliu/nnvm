/*!
 *  Copyright (c) 2018 by Contributors
 * \file batch_norm_inference.h
 * \brief
 */
#ifndef NNVM_TOP_CONTRIB_BATCH_NORM_INFERENCE_H
#define NNVM_TOP_CONTRIB_BATCH_NORM_INFERENCE_H

#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <nnvm/tuple.h>

namespace nnvm {
namespace top {
namespace contrib {

struct BatchNormInferenceParam : public dmlc::Parameter<BatchNormInferenceParam> {
  double epsilon;
  bool center;
  bool scale;

  DMLC_DECLARE_PARAMETER(BatchNormInferenceParam) {
  DMLC_DECLARE_FIELD(epsilon).set_default(1e-5)
  .describe("Small float added to variance to avoid dividing by zero.");
  DMLC_DECLARE_FIELD(center).set_default(true)
  .describe("If True, add offset of `beta` to normalized tensor."
  "If False, `beta` is ignored.");
  DMLC_DECLARE_FIELD(scale).set_default(true)
  .describe("If True, multiply by `gamma`. If False, `gamma` is not used."
  "When the next layer is piecewise linear (also e.g. `nn.relu`),"
  "this can be disabled since the scaling"
  "will be done by the next layer.");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kGamma = 1;
  static const constexpr int kBeta = 2;
  static const constexpr int kMovingMean = 3;
  static const constexpr int kMovingVariance = 4;
};

}  // namespace contrib
}  // namespace top
}  // namspace nnvm

#endif // NNVM_TOP_CONTRIB_BATCH_NORM_INFERENCE_H
