/*!
 *  Copyright (c) 2017 by Contributors
 * \file layout.h
 * \brief Layout expression.
 */
#ifndef NNVM_COMPILER_LAYOUT_H_
#define NNVM_COMPILER_LAYOUT_H_

#include <dmlc/parameter.h>
#include <string>
#include <vector>

namespace nnvm {

class Layout {
 public:
  Layout(const std::string& layout)
    : layout(layout), major_position(26, -1), minor_position(26, -1), minor_factor(26, 0) {
    uint32_t factor = 0;
    uint32_t curr = 0;
    for (size_t i = 0; i < layout.size(); ++i) {
      const char c = layout.at(i);
      if (IsMajorAxis(c)) {
        int pos = c - 'A';
        CHECK_EQ(factor, 0) << "Invalid layout " << layout
                            << ": invalid factor size " << factor << " before axis " << c;
        CHECK_EQ(major_position[pos], -1) << "Invalid layout " << layout
                                          << ": duplicate axis " << c;
        major_position[pos] = curr++;
        layout_simplified.push_back(c);
      } else if (IsMinorAxis(c)) {
        int pos = c - 'a';
        CHECK_GT(factor, 0) << "Invalid layout " << layout
                            << ": invalid factor size " << factor << " for axis " << c;
        CHECK_EQ(minor_position[pos], -1) << "Invalid layout " << layout
                                          << ": duplicate axis " << c;
        CHECK_EQ(minor_factor[pos], 0) << "Invalid layout " << layout
                                       << ": duplicate axis " << c;
        minor_position[pos] = curr++;
        minor_factor[pos] = factor;
        layout_simplified.push_back(c);
        factor = 0;
      } else if (c >= '0' && c <= '9') {
        factor = factor * 10 + c - '0';
      } else {
        LOG(FATAL) << "Invalid layout " << layout;
      }
    }
    for (char axis : layout_simplified) {
      CHECK(IsMajorAxis(axis) || major_position[axis-'a'] >= 0)
        << "Invalid layout " << layout << ": missing axis "
        << static_cast<char>(axis - 'a' + 'A');
    }
  }

  static inline bool IsMajorAxis(char c) {
    return c >= 'A' && c <= 'Z';
  }

  static inline bool IsMinorAxis(char c) {
    return c >= 'a' && c <= 'z';
  }

  inline bool ConvertibleTo(const Layout &dst);

  using iterator = std::vector<char>::const_iterator;
  using reverse_iterator = std::vector<char>::const_reverse_iterator;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(layout_simplified.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(layout_simplified.end());
  }
  /*! \return rbegin iterator */
  inline reverse_iterator rbegin() const {
    return reverse_iterator(layout_simplified.rbegin());
  }
  /*! \return rend iterator */
  inline reverse_iterator rend() const {
    return reverse_iterator(layout_simplified.rend());
  }

  inline size_t size() const {
    return layout_simplified.size();
  }

  inline int PosMajor(char c) const {
    CHECK(IsMajorAxis(c) || IsMinorAxis(c)) << "Invalid axis " << c;
    char idx = IsMajorAxis(c) ? c - 'A' : c - 'a';
    return major_position[idx];
  }

  inline int PosMinor(char c) const {
    CHECK(IsMajorAxis(c) || IsMinorAxis(c)) << "Invalid axis " << c;
    char idx = IsMajorAxis(c) ? c - 'A' : c - 'a';
    return minor_position[idx];
  }

  inline uint32_t FactorSize(char axis) const {
    CHECK(IsMajorAxis(axis) || IsMinorAxis(axis)) << "Invalid axis " << axis;
    char idx = IsMajorAxis(axis) ? axis - 'A' : axis - 'a';
    return minor_factor[idx];
  }

 private:
  const std::string layout;
  std::vector<int> major_position;
  std::vector<int> minor_position;
  std::vector<uint32_t> minor_factor;
  std::vector<char> layout_simplified;

};

inline bool Layout::ConvertibleTo(const Layout &dst) {
  for (size_t i = 0; i < major_position.size(); ++i) {
    if ((major_position[i] >= 0 && dst.major_position[i] < 0) ||
        (major_position[i] < 0 && dst.major_position[i] >= 0)) {
      return false;
    }
  }
  return true;
}


}  // namespace nnvm

#endif  // NNVM_COMPILER_LAYOUT_H_
