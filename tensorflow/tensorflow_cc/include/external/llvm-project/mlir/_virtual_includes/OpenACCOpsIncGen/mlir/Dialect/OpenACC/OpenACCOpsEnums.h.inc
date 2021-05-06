/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
// default attribute values
enum class DefaultAttr {
  none,
  present,
};

::llvm::StringRef stringifyDefaultAttr(DefaultAttr);
::llvm::Optional<DefaultAttr> symbolizeDefaultAttr(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(DefaultAttr enumValue) {
  return stringifyDefaultAttr(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DefaultAttr> symbolizeEnum<DefaultAttr>(::llvm::StringRef str) {
  return symbolizeDefaultAttr(str);
}
} // namespace acc
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::DefaultAttr> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::acc::DefaultAttr>::type>;

  static inline ::mlir::acc::DefaultAttr getEmptyKey() {
    return static_cast<::mlir::acc::DefaultAttr>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::DefaultAttr getTombstoneKey() {
    return static_cast<::mlir::acc::DefaultAttr>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::DefaultAttr &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::acc::DefaultAttr>::type>(val));
  }

  static bool isEqual(const ::mlir::acc::DefaultAttr &lhs, const ::mlir::acc::DefaultAttr &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace acc {
// built-in reduction operations supported by OpenACC
enum class ReductionOpAttr {
  redop_add,
  redop_mul,
  redop_max,
  redop_min,
  redop_and,
  redop_or,
  redop_xor,
  redop_leqv,
  redop_lneqv,
  redop_land,
  redop_lor,
};

::llvm::StringRef stringifyReductionOpAttr(ReductionOpAttr);
::llvm::Optional<ReductionOpAttr> symbolizeReductionOpAttr(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ReductionOpAttr enumValue) {
  return stringifyReductionOpAttr(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ReductionOpAttr> symbolizeEnum<ReductionOpAttr>(::llvm::StringRef str) {
  return symbolizeReductionOpAttr(str);
}
} // namespace acc
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ReductionOpAttr> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::acc::ReductionOpAttr>::type>;

  static inline ::mlir::acc::ReductionOpAttr getEmptyKey() {
    return static_cast<::mlir::acc::ReductionOpAttr>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ReductionOpAttr getTombstoneKey() {
    return static_cast<::mlir::acc::ReductionOpAttr>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ReductionOpAttr &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::acc::ReductionOpAttr>::type>(val));
  }

  static bool isEqual(const ::mlir::acc::ReductionOpAttr &lhs, const ::mlir::acc::ReductionOpAttr &rhs) {
    return lhs == rhs;
  }
};
}

