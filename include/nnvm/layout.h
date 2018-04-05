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
#include <algorithm>

namespace nnvm {

class Layout {
 public:
  using LayoutAxis = char;

  Layout() : name_("__undef__") {} // NOLINT(*)

  inline Layout(const std::string& layout) { // NOLINT(*)
    parse(layout);
  }
  /*!
   * \brief copy constructor from another layout
   * \param s the source layout
   */
  inline Layout(const Layout& s) { // NOLINT(*)
    this->parse(s.name_);
  }
  /*!
   * \brief move constructor from Layout
   * \param src the source layout
   */
  inline Layout(Layout&& src) { // NOLINT(*)
    this->swap(src);
  }
  /*!
   * \brief assignment from another layout.
   * \param src source layout
   * \return reference of self
   */
  inline Layout& operator=(const Layout& src) {
    this->parse(src.name_);
    return *this;
  }
  /*!
   * \brief assignment from rvalue of another layout.
   * \param src source layout
   * \return reference of self
   */
  inline Layout& operator=(Layout&& src) {
    Layout(std::move(src)).swap(*this);
    return *this;
  }
  /*!
   * \return whether two layout equals
   * \param s the layout to compare against
   */
  inline bool operator==(const Layout& s) const {
    return name_ == s.name_;
  }
  /*!
   * \return whether two layout not equal
   * \param s the layout to compare against
   */
  inline bool operator!=(const Layout& s) const {
    return !(*this == s);
  }

  static inline bool IsMajorAxis(LayoutAxis c) {
    return c >= 'A' && c <= 'Z';
  }

  static inline bool IsMinorAxis(LayoutAxis c) {
    return c >= 'a' && c <= 'z';
  }

  static inline const Layout& Undef() {
    static Layout undef;
    return undef;
  }

  void parse(const std::string& layout) {
    name_ = layout;
    if (layout == "__undef__") return;

    std::fill_n(major_position_, kUniqueAxis, -1);
    std::fill_n(minor_position_, kUniqueAxis, -1);
    std::fill_n(minor_factor_, kUniqueAxis, 0);

    uint32_t factor = 0;
    uint32_t curr = 0;
    for (size_t i = 0; i < layout.size(); ++i) {
      const LayoutAxis c = layout.at(i);
      if (IsMajorAxis(c)) {
        int pos = c - 'A';
        CHECK_EQ(factor, 0) << "Invalid layout " << layout
                            << ": invalid factor size " << factor << " before axis " << c;
        CHECK_EQ(major_position_[pos], -1) << "Invalid layout " << layout
                                           << ": duplicate axis " << c;
        major_position_[pos] = curr++;
        layout_simplified_.push_back(c);
      } else if (IsMinorAxis(c)) {
        int pos = c - 'a';
        CHECK_GT(factor, 0) << "Invalid layout " << layout
                            << ": invalid factor size " << factor << " for axis " << c;
        CHECK_EQ(minor_position_[pos], -1) << "Invalid layout " << layout
                                           << ": duplicate axis " << c;
        CHECK_EQ(minor_factor_[pos], 0) << "Invalid layout " << layout
                                        << ": duplicate axis " << c;
        minor_position_[pos] = curr++;
        minor_factor_[pos] = factor;
        layout_simplified_.push_back(c);
        factor = 0;
      } else if (c >= '0' && c <= '9') {
        factor = factor * 10 + c - '0';
      } else {
        LOG(FATAL) << "Invalid layout " << layout;
      }
    }
    CHECK(!layout_simplified_.empty()) << "Invalid layout " << layout;
    for (LayoutAxis axis : layout_simplified_) {
      CHECK(IsMajorAxis(axis) || major_position_[axis-'a'] >= 0)
        << "Invalid layout " << layout << ": missing axis "
        << static_cast<char>(axis - 'a' + 'A');
    }
  }

  /*!
   * \brief Swap current object with other
   * \param other another object to be swapped.
   */
  inline void swap(Layout& other) {  // NOLINT(*)
    std::swap(name_, other.name_);
    std::swap(major_position_, other.major_position_);
    std::swap(minor_position_, other.minor_position_);
    std::swap(minor_factor_, other.minor_factor_);
  }

  inline bool Convertible(const Layout &dst) const {
    for (size_t i = 0; i < kUniqueAxis; ++i) {
      if ((major_position_[i] >= 0 && dst.major_position_[i] < 0) ||
          (major_position_[i] < 0 && dst.major_position_[i] >= 0)) {
        return false;
      }
    }
    return true;
  }

  using iterator = std::vector<LayoutAxis>::const_iterator;
  using reverse_iterator = std::vector<LayoutAxis>::const_reverse_iterator;

  /*! \return begin iterator */
  inline iterator begin() const {
    return layout_simplified_.begin();
  }
  /*! \return end iterator */
  inline iterator end() const {
    return layout_simplified_.end();
  }
  /*! \return rbegin iterator */
  inline reverse_iterator rbegin() const {
    return layout_simplified_.rbegin();
  }
  /*! \return rend iterator */
  inline reverse_iterator rend() const {
    return layout_simplified_.rend();
  }

  inline size_t ndim() const {
    return layout_simplified_.size();
  }

  inline int PosMajor(LayoutAxis c) const {
    CHECK(IsMajorAxis(c) || IsMinorAxis(c)) << "Invalid axis " << c;
    int idx = IsMajorAxis(c) ? c - 'A' : c - 'a';
    return major_position_[idx];
  }

  inline int PosMinor(LayoutAxis c) const {
    CHECK(IsMajorAxis(c) || IsMinorAxis(c)) << "Invalid axis " << c;
    int idx = IsMajorAxis(c) ? c - 'A' : c - 'a';
    return minor_position_[idx];
  }

  inline uint32_t FactorSize(LayoutAxis axis) const {
    CHECK(IsMajorAxis(axis) || IsMinorAxis(axis)) << "Invalid axis " << axis;
    int idx = IsMajorAxis(axis) ? axis - 'A' : axis - 'a';
    return minor_factor_[idx];
  }

  inline const LayoutAxis operator[](size_t i) const {
    return layout_simplified_[i];
  }

  inline bool IsDefined() const {
    return name_ != "__undef__";
  }

  inline const std::string& name() const {
    return name_;
  }

  inline void Save(dmlc::JSONWriter* writer) const {
    writer->Write(name_);
  }

  /*!
   * \brief Load layout from JSON.
   * \param reader JSONReader
   */
  inline void Load(dmlc::JSONReader* reader) {
    std::string tmp;
    reader->Read(&tmp);
    this->parse(tmp);
  }

  /*!
   * \brief allow output string of layout to ostream
   * \param os the output stream
   * \param l the layout
   * \return the ostream
   */
  friend std::ostream& operator<<(std::ostream& os, const Layout& l) {
    os << l.name_;
    return os;
  }

 private:
  static const uint32_t kUniqueAxis = 26;

  std::string name_;
  int major_position_[kUniqueAxis];
  int minor_position_[kUniqueAxis];
  uint32_t minor_factor_[kUniqueAxis];
  std::vector<LayoutAxis> layout_simplified_;
};

}  // namespace nnvm

#endif  // NNVM_COMPILER_LAYOUT_H_
