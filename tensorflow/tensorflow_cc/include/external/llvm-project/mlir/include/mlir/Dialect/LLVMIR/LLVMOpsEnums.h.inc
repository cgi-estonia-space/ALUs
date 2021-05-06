/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
// llvm.atomicrmw binary operations
enum class AtomicBinOp : uint64_t {
  xchg = 0,
  add = 1,
  sub = 2,
  _and = 3,
  nand = 4,
  _or = 5,
  _xor = 6,
  max = 7,
  min = 8,
  umax = 9,
  umin = 10,
  fadd = 11,
  fsub = 12,
};

::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(uint64_t);
::llvm::StringRef stringifyAtomicBinOp(AtomicBinOp);
::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicBinOp() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(AtomicBinOp enumValue) {
  return stringifyAtomicBinOp(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AtomicBinOp> symbolizeEnum<AtomicBinOp>(::llvm::StringRef str) {
  return symbolizeAtomicBinOp(str);
}
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AtomicBinOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AtomicBinOp getEmptyKey() {
    return static_cast<::mlir::LLVM::AtomicBinOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AtomicBinOp getTombstoneKey() {
    return static_cast<::mlir::LLVM::AtomicBinOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AtomicBinOp &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AtomicBinOp &lhs, const ::mlir::LLVM::AtomicBinOp &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// Atomic ordering for LLVM's memory model
enum class AtomicOrdering : uint64_t {
  not_atomic = 0,
  unordered = 1,
  monotonic = 2,
  acquire = 4,
  release = 5,
  acq_rel = 6,
  seq_cst = 7,
};

::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(uint64_t);
::llvm::StringRef stringifyAtomicOrdering(AtomicOrdering);
::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAtomicOrdering() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(AtomicOrdering enumValue) {
  return stringifyAtomicOrdering(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<AtomicOrdering> symbolizeEnum<AtomicOrdering>(::llvm::StringRef str) {
  return symbolizeAtomicOrdering(str);
}
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::AtomicOrdering> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::AtomicOrdering getEmptyKey() {
    return static_cast<::mlir::LLVM::AtomicOrdering>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::AtomicOrdering getTombstoneKey() {
    return static_cast<::mlir::LLVM::AtomicOrdering>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::AtomicOrdering &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::AtomicOrdering &lhs, const ::mlir::LLVM::AtomicOrdering &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// llvm.fcmp comparison predicate
enum class FCmpPredicate : uint64_t {
  _false = 0,
  oeq = 1,
  ogt = 2,
  oge = 3,
  olt = 4,
  ole = 5,
  one = 6,
  ord = 7,
  ueq = 8,
  ugt = 9,
  uge = 10,
  ult = 11,
  ule = 12,
  une = 13,
  uno = 14,
  _true = 15,
};

::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(uint64_t);
::llvm::StringRef stringifyFCmpPredicate(FCmpPredicate);
::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFCmpPredicate() {
  return 15;
}


inline ::llvm::StringRef stringifyEnum(FCmpPredicate enumValue) {
  return stringifyFCmpPredicate(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FCmpPredicate> symbolizeEnum<FCmpPredicate>(::llvm::StringRef str) {
  return symbolizeFCmpPredicate(str);
}
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::FCmpPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::FCmpPredicate getEmptyKey() {
    return static_cast<::mlir::LLVM::FCmpPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::FCmpPredicate getTombstoneKey() {
    return static_cast<::mlir::LLVM::FCmpPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::FCmpPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::FCmpPredicate &lhs, const ::mlir::LLVM::FCmpPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// llvm.icmp comparison predicate
enum class ICmpPredicate : uint64_t {
  eq = 0,
  ne = 1,
  slt = 2,
  sle = 3,
  sgt = 4,
  sge = 5,
  ult = 6,
  ule = 7,
  ugt = 8,
  uge = 9,
};

::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(uint64_t);
::llvm::StringRef stringifyICmpPredicate(ICmpPredicate);
::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForICmpPredicate() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(ICmpPredicate enumValue) {
  return stringifyICmpPredicate(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<ICmpPredicate> symbolizeEnum<ICmpPredicate>(::llvm::StringRef str) {
  return symbolizeICmpPredicate(str);
}
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::ICmpPredicate> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::ICmpPredicate getEmptyKey() {
    return static_cast<::mlir::LLVM::ICmpPredicate>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::ICmpPredicate getTombstoneKey() {
    return static_cast<::mlir::LLVM::ICmpPredicate>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::ICmpPredicate &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::ICmpPredicate &lhs, const ::mlir::LLVM::ICmpPredicate &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace LLVM {
// LLVM linkage types
enum class Linkage : uint64_t {
  Private = 0,
  Internal = 1,
  AvailableExternally = 2,
  Linkonce = 3,
  Weak = 4,
  Common = 5,
  Appending = 6,
  ExternWeak = 7,
  LinkonceODR = 8,
  WeakODR = 9,
  External = 10,
};

::llvm::Optional<Linkage> symbolizeLinkage(uint64_t);
::llvm::StringRef stringifyLinkage(Linkage);
::llvm::Optional<Linkage> symbolizeLinkage(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLinkage() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(Linkage enumValue) {
  return stringifyLinkage(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<Linkage> symbolizeEnum<Linkage>(::llvm::StringRef str) {
  return symbolizeLinkage(str);
}
} // namespace LLVM
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::LLVM::Linkage> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::LLVM::Linkage getEmptyKey() {
    return static_cast<::mlir::LLVM::Linkage>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::LLVM::Linkage getTombstoneKey() {
    return static_cast<::mlir::LLVM::Linkage>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::LLVM::Linkage &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::LLVM::Linkage &lhs, const ::mlir::LLVM::Linkage &rhs) {
    return lhs == rhs;
  }
};
}

