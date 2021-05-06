/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo_gpu {

// GPU Convolution backend configuration
class ConvolutionBackendConfig : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ConvolutionBackendConfig get(
      ::mlir::IntegerAttr algorithm,
      ::mlir::BoolAttr tensor_ops_enabled,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr algorithm() const;
  ::mlir::BoolAttr tensor_ops_enabled() const;
};

} // namespace mlir
} // namespace lmhlo_gpu
