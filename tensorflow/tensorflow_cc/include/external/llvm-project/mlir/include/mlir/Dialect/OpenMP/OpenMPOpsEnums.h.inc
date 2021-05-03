/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
// default clause
enum class ClauseDefault {
  defprivate,
  deffirstprivate,
  defshared,
  defnone,
};

::llvm::StringRef stringifyClauseDefault(ClauseDefault);
::llvm::Optional<ClauseDefault> symbolizeClauseDefault(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ClauseDefault enumValue) {
  return stringifyClauseDefault(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseDefault> symbolizeEnum<ClauseDefault>(::llvm::StringRef str) {
  return symbolizeClauseDefault(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseDefault> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::omp::ClauseDefault>::type>;

  static inline ::mlir::omp::ClauseDefault getEmptyKey() {
    return static_cast<::mlir::omp::ClauseDefault>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseDefault getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseDefault>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseDefault &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::omp::ClauseDefault>::type>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseDefault &lhs, const ::mlir::omp::ClauseDefault &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace omp {
// ProcBindKind Clause
enum class ClauseProcBindKind {
  master,
  close,
  spread,
};

::llvm::StringRef stringifyClauseProcBindKind(ClauseProcBindKind);
::llvm::Optional<ClauseProcBindKind> symbolizeClauseProcBindKind(::llvm::StringRef);

inline ::llvm::StringRef stringifyEnum(ClauseProcBindKind enumValue) {
  return stringifyClauseProcBindKind(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ClauseProcBindKind> symbolizeEnum<ClauseProcBindKind>(::llvm::StringRef str) {
  return symbolizeClauseProcBindKind(str);
}
} // namespace omp
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::omp::ClauseProcBindKind> {
  using StorageInfo = ::llvm::DenseMapInfo<std::underlying_type<::mlir::omp::ClauseProcBindKind>::type>;

  static inline ::mlir::omp::ClauseProcBindKind getEmptyKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::omp::ClauseProcBindKind getTombstoneKey() {
    return static_cast<::mlir::omp::ClauseProcBindKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::omp::ClauseProcBindKind &val) {
    return StorageInfo::getHashValue(static_cast<std::underlying_type<::mlir::omp::ClauseProcBindKind>::type>(val));
  }

  static bool isEqual(const ::mlir::omp::ClauseProcBindKind &lhs, const ::mlir::omp::ClauseProcBindKind &rhs) {
    return lhs == rhs;
  }
};
}

