/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace ROCDL {

class ROCDLDialect : public ::mlir::Dialect {
  explicit ROCDLDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<ROCDLDialect>()) {
    
    getContext()->getOrLoadDialect<LLVM::LLVMDialect>();

    initialize();
  }
  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static ::llvm::StringRef getDialectNamespace() { return "rocdl"; }
};
} // namespace ROCDL
} // namespace mlir
