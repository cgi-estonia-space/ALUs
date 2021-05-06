/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace pdl_interp {

class PDLInterpDialect : public ::mlir::Dialect {
  explicit PDLInterpDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<PDLInterpDialect>()) {
    
    getContext()->getOrLoadDialect<pdl::PDLDialect>();

    initialize();
  }
  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static ::llvm::StringRef getDialectNamespace() { return "pdl_interp"; }
};
} // namespace pdl_interp
} // namespace mlir
