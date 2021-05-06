/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicBinOp(AtomicBinOp val) {
  switch (val) {
    case AtomicBinOp::xchg: return "xchg";
    case AtomicBinOp::add: return "add";
    case AtomicBinOp::sub: return "sub";
    case AtomicBinOp::_and: return "_and";
    case AtomicBinOp::nand: return "nand";
    case AtomicBinOp::_or: return "_or";
    case AtomicBinOp::_xor: return "_xor";
    case AtomicBinOp::max: return "max";
    case AtomicBinOp::min: return "min";
    case AtomicBinOp::umax: return "umax";
    case AtomicBinOp::umin: return "umin";
    case AtomicBinOp::fadd: return "fadd";
    case AtomicBinOp::fsub: return "fsub";
  }
  return "";
}

::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AtomicBinOp>>(str)
      .Case("xchg", AtomicBinOp::xchg)
      .Case("add", AtomicBinOp::add)
      .Case("sub", AtomicBinOp::sub)
      .Case("_and", AtomicBinOp::_and)
      .Case("nand", AtomicBinOp::nand)
      .Case("_or", AtomicBinOp::_or)
      .Case("_xor", AtomicBinOp::_xor)
      .Case("max", AtomicBinOp::max)
      .Case("min", AtomicBinOp::min)
      .Case("umax", AtomicBinOp::umax)
      .Case("umin", AtomicBinOp::umin)
      .Case("fadd", AtomicBinOp::fadd)
      .Case("fsub", AtomicBinOp::fsub)
      .Default(::llvm::None);
}
::llvm::Optional<AtomicBinOp> symbolizeAtomicBinOp(uint64_t value) {
  switch (value) {
  case 0: return AtomicBinOp::xchg;
  case 1: return AtomicBinOp::add;
  case 2: return AtomicBinOp::sub;
  case 3: return AtomicBinOp::_and;
  case 4: return AtomicBinOp::nand;
  case 5: return AtomicBinOp::_or;
  case 6: return AtomicBinOp::_xor;
  case 7: return AtomicBinOp::max;
  case 8: return AtomicBinOp::min;
  case 9: return AtomicBinOp::umax;
  case 10: return AtomicBinOp::umin;
  case 11: return AtomicBinOp::fadd;
  case 12: return AtomicBinOp::fsub;
  default: return ::llvm::None;
  }
}

} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyAtomicOrdering(AtomicOrdering val) {
  switch (val) {
    case AtomicOrdering::not_atomic: return "not_atomic";
    case AtomicOrdering::unordered: return "unordered";
    case AtomicOrdering::monotonic: return "monotonic";
    case AtomicOrdering::acquire: return "acquire";
    case AtomicOrdering::release: return "release";
    case AtomicOrdering::acq_rel: return "acq_rel";
    case AtomicOrdering::seq_cst: return "seq_cst";
  }
  return "";
}

::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AtomicOrdering>>(str)
      .Case("not_atomic", AtomicOrdering::not_atomic)
      .Case("unordered", AtomicOrdering::unordered)
      .Case("monotonic", AtomicOrdering::monotonic)
      .Case("acquire", AtomicOrdering::acquire)
      .Case("release", AtomicOrdering::release)
      .Case("acq_rel", AtomicOrdering::acq_rel)
      .Case("seq_cst", AtomicOrdering::seq_cst)
      .Default(::llvm::None);
}
::llvm::Optional<AtomicOrdering> symbolizeAtomicOrdering(uint64_t value) {
  switch (value) {
  case 0: return AtomicOrdering::not_atomic;
  case 1: return AtomicOrdering::unordered;
  case 2: return AtomicOrdering::monotonic;
  case 4: return AtomicOrdering::acquire;
  case 5: return AtomicOrdering::release;
  case 6: return AtomicOrdering::acq_rel;
  case 7: return AtomicOrdering::seq_cst;
  default: return ::llvm::None;
  }
}

} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyFCmpPredicate(FCmpPredicate val) {
  switch (val) {
    case FCmpPredicate::_false: return "_false";
    case FCmpPredicate::oeq: return "oeq";
    case FCmpPredicate::ogt: return "ogt";
    case FCmpPredicate::oge: return "oge";
    case FCmpPredicate::olt: return "olt";
    case FCmpPredicate::ole: return "ole";
    case FCmpPredicate::one: return "one";
    case FCmpPredicate::ord: return "ord";
    case FCmpPredicate::ueq: return "ueq";
    case FCmpPredicate::ugt: return "ugt";
    case FCmpPredicate::uge: return "uge";
    case FCmpPredicate::ult: return "ult";
    case FCmpPredicate::ule: return "ule";
    case FCmpPredicate::une: return "une";
    case FCmpPredicate::uno: return "uno";
    case FCmpPredicate::_true: return "_true";
  }
  return "";
}

::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<FCmpPredicate>>(str)
      .Case("_false", FCmpPredicate::_false)
      .Case("oeq", FCmpPredicate::oeq)
      .Case("ogt", FCmpPredicate::ogt)
      .Case("oge", FCmpPredicate::oge)
      .Case("olt", FCmpPredicate::olt)
      .Case("ole", FCmpPredicate::ole)
      .Case("one", FCmpPredicate::one)
      .Case("ord", FCmpPredicate::ord)
      .Case("ueq", FCmpPredicate::ueq)
      .Case("ugt", FCmpPredicate::ugt)
      .Case("uge", FCmpPredicate::uge)
      .Case("ult", FCmpPredicate::ult)
      .Case("ule", FCmpPredicate::ule)
      .Case("une", FCmpPredicate::une)
      .Case("uno", FCmpPredicate::uno)
      .Case("_true", FCmpPredicate::_true)
      .Default(::llvm::None);
}
::llvm::Optional<FCmpPredicate> symbolizeFCmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return FCmpPredicate::_false;
  case 1: return FCmpPredicate::oeq;
  case 2: return FCmpPredicate::ogt;
  case 3: return FCmpPredicate::oge;
  case 4: return FCmpPredicate::olt;
  case 5: return FCmpPredicate::ole;
  case 6: return FCmpPredicate::one;
  case 7: return FCmpPredicate::ord;
  case 8: return FCmpPredicate::ueq;
  case 9: return FCmpPredicate::ugt;
  case 10: return FCmpPredicate::uge;
  case 11: return FCmpPredicate::ult;
  case 12: return FCmpPredicate::ule;
  case 13: return FCmpPredicate::une;
  case 14: return FCmpPredicate::uno;
  case 15: return FCmpPredicate::_true;
  default: return ::llvm::None;
  }
}

} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyICmpPredicate(ICmpPredicate val) {
  switch (val) {
    case ICmpPredicate::eq: return "eq";
    case ICmpPredicate::ne: return "ne";
    case ICmpPredicate::slt: return "slt";
    case ICmpPredicate::sle: return "sle";
    case ICmpPredicate::sgt: return "sgt";
    case ICmpPredicate::sge: return "sge";
    case ICmpPredicate::ult: return "ult";
    case ICmpPredicate::ule: return "ule";
    case ICmpPredicate::ugt: return "ugt";
    case ICmpPredicate::uge: return "uge";
  }
  return "";
}

::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ICmpPredicate>>(str)
      .Case("eq", ICmpPredicate::eq)
      .Case("ne", ICmpPredicate::ne)
      .Case("slt", ICmpPredicate::slt)
      .Case("sle", ICmpPredicate::sle)
      .Case("sgt", ICmpPredicate::sgt)
      .Case("sge", ICmpPredicate::sge)
      .Case("ult", ICmpPredicate::ult)
      .Case("ule", ICmpPredicate::ule)
      .Case("ugt", ICmpPredicate::ugt)
      .Case("uge", ICmpPredicate::uge)
      .Default(::llvm::None);
}
::llvm::Optional<ICmpPredicate> symbolizeICmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return ICmpPredicate::eq;
  case 1: return ICmpPredicate::ne;
  case 2: return ICmpPredicate::slt;
  case 3: return ICmpPredicate::sle;
  case 4: return ICmpPredicate::sgt;
  case 5: return ICmpPredicate::sge;
  case 6: return ICmpPredicate::ult;
  case 7: return ICmpPredicate::ule;
  case 8: return ICmpPredicate::ugt;
  case 9: return ICmpPredicate::uge;
  default: return ::llvm::None;
  }
}

} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
::llvm::StringRef stringifyLinkage(Linkage val) {
  switch (val) {
    case Linkage::Private: return "private";
    case Linkage::Internal: return "internal";
    case Linkage::AvailableExternally: return "available_externally";
    case Linkage::Linkonce: return "linkonce";
    case Linkage::Weak: return "weak";
    case Linkage::Common: return "common";
    case Linkage::Appending: return "appending";
    case Linkage::ExternWeak: return "extern_weak";
    case Linkage::LinkonceODR: return "linkonce_odr";
    case Linkage::WeakODR: return "weak_odr";
    case Linkage::External: return "external";
  }
  return "";
}

::llvm::Optional<Linkage> symbolizeLinkage(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Linkage>>(str)
      .Case("private", Linkage::Private)
      .Case("internal", Linkage::Internal)
      .Case("available_externally", Linkage::AvailableExternally)
      .Case("linkonce", Linkage::Linkonce)
      .Case("weak", Linkage::Weak)
      .Case("common", Linkage::Common)
      .Case("appending", Linkage::Appending)
      .Case("extern_weak", Linkage::ExternWeak)
      .Case("linkonce_odr", Linkage::LinkonceODR)
      .Case("weak_odr", Linkage::WeakODR)
      .Case("external", Linkage::External)
      .Default(::llvm::None);
}
::llvm::Optional<Linkage> symbolizeLinkage(uint64_t value) {
  switch (value) {
  case 0: return Linkage::Private;
  case 1: return Linkage::Internal;
  case 2: return Linkage::AvailableExternally;
  case 3: return Linkage::Linkonce;
  case 4: return Linkage::Weak;
  case 5: return Linkage::Common;
  case 6: return Linkage::Appending;
  case 7: return Linkage::ExternWeak;
  case 8: return Linkage::LinkonceODR;
  case 9: return Linkage::WeakODR;
  case 10: return Linkage::External;
  default: return ::llvm::None;
  }
}

} // namespace LLVM
} // namespace mlir

