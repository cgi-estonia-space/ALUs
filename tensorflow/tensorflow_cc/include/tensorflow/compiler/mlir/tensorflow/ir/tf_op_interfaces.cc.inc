/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

Optional<ContractionFusion> ContractionFusableInterface::GetContractionFusion() {
      return getImpl()->GetContractionFusion(getOperation());
  }
SmallVector<unsigned, 4> FoldOperandsTransposeInterface::GetLayoutDependentArgs() {
      return getImpl()->GetLayoutDependentArgs(getOperation());
  }
SmallVector<unsigned, 4> FoldOperandsTransposeInterface::GetLayoutDependentResults() {
      return getImpl()->GetLayoutDependentResults(getOperation());
  }
LogicalResult FoldOperandsTransposeInterface::FoldOperandsPermutation(ArrayRef<int64_t> permutation) {
      return getImpl()->FoldOperandsPermutation(getOperation(), permutation);
  }
StringRef LayoutSensitiveInterface::data_format() {
      return getImpl()->data_format(getOperation());
  }
SmallVector<unsigned, 4> LayoutSensitiveInterface::GetLayoutDependentArgs() {
      return getImpl()->GetLayoutDependentArgs(getOperation());
  }
SmallVector<unsigned, 4> LayoutSensitiveInterface::GetLayoutDependentResults() {
      return getImpl()->GetLayoutDependentResults(getOperation());
  }
StringRef LayoutSensitiveInterface::GetOptimalLayout(const RuntimeDevices& devices) {
      return getImpl()->GetOptimalLayout(getOperation(), devices);
  }
LogicalResult LayoutSensitiveInterface::UpdateDataFormat(StringRef data_format) {
      return getImpl()->UpdateDataFormat(getOperation(), data_format);
  }
ResourceHandleValueAndId ResourceHandleAllocatorInterface::GetResourceHandleValueAndId(llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) {
      return getImpl()->GetResourceHandleValueAndId(getOperation(), resource_handle_id_map, next_id);
  }
