/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Capability Implication                                              *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

ArrayRef<Capability> spirv::getDirectImpliedCapabilities(Capability cap) {
  switch (cap) {
  default: return {};
  case Capability::Shader: {static const Capability implies[1] = {Capability::Matrix}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Vector16: {static const Capability implies[1] = {Capability::Kernel}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Float16Buffer: {static const Capability implies[1] = {Capability::Kernel}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Int64Atomics: {static const Capability implies[1] = {Capability::Int64}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageBasic: {static const Capability implies[1] = {Capability::Kernel}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Pipes: {static const Capability implies[1] = {Capability::Kernel}; return ArrayRef<Capability>(implies, 1); }
  case Capability::DeviceEnqueue: {static const Capability implies[1] = {Capability::Kernel}; return ArrayRef<Capability>(implies, 1); }
  case Capability::LiteralSampler: {static const Capability implies[1] = {Capability::Kernel}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GenericPointer: {static const Capability implies[1] = {Capability::Addresses}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Image1D: {static const Capability implies[1] = {Capability::Sampled1D}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageBuffer: {static const Capability implies[1] = {Capability::SampledBuffer}; return ArrayRef<Capability>(implies, 1); }
  case Capability::NamedBarrier: {static const Capability implies[1] = {Capability::Kernel}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GroupNonUniformVote: {static const Capability implies[1] = {Capability::GroupNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GroupNonUniformArithmetic: {static const Capability implies[1] = {Capability::GroupNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GroupNonUniformBallot: {static const Capability implies[1] = {Capability::GroupNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GroupNonUniformShuffle: {static const Capability implies[1] = {Capability::GroupNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GroupNonUniformShuffleRelative: {static const Capability implies[1] = {Capability::GroupNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GroupNonUniformClustered: {static const Capability implies[1] = {Capability::GroupNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GroupNonUniformQuad: {static const Capability implies[1] = {Capability::GroupNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageUniform16: {static const Capability implies[1] = {Capability::StorageBuffer16BitAccess}; return ArrayRef<Capability>(implies, 1); }
  case Capability::UniformAndStorageBuffer8BitAccess: {static const Capability implies[1] = {Capability::StorageBuffer8BitAccess}; return ArrayRef<Capability>(implies, 1); }
  case Capability::UniformTexelBufferArrayDynamicIndexing: {static const Capability implies[1] = {Capability::SampledBuffer}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Geometry: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Tessellation: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageReadWrite: {static const Capability implies[1] = {Capability::ImageBasic}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageMipmap: {static const Capability implies[1] = {Capability::ImageBasic}; return ArrayRef<Capability>(implies, 1); }
  case Capability::AtomicStorage: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageGatherExtended: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageImageMultisample: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::UniformBufferArrayDynamicIndexing: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SampledImageArrayDynamicIndexing: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageBufferArrayDynamicIndexing: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageImageArrayDynamicIndexing: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ClipDistance: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::CullDistance: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SampleRateShading: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SampledRect: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::InputAttachment: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SparseResidency: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::MinLod: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SampledCubeArray: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageMSArray: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageImageExtendedFormats: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageQuery: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::DerivativeControl: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::InterpolationFunction: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::TransformFeedback: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageImageReadWithoutFormat: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageImageWriteWithoutFormat: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SubgroupDispatch: {static const Capability implies[1] = {Capability::DeviceEnqueue}; return ArrayRef<Capability>(implies, 1); }
  case Capability::PipeStorage: {static const Capability implies[1] = {Capability::Pipes}; return ArrayRef<Capability>(implies, 1); }
  case Capability::DrawParameters: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::MultiView: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::VariablePointersStorageBuffer: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::Float16ImageAMD: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageGatherBiasLodAMD: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::FragmentMaskAMD: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StencilExportEXT: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageReadWriteLodAMD: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ShaderClockKHR: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::FragmentFullyCoveredEXT: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::MeshShadingNV: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::FragmentDensityEXT: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ShaderNonUniform: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::RuntimeDescriptorArray: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageTexelBufferArrayDynamicIndexing: {static const Capability implies[1] = {Capability::ImageBuffer}; return ArrayRef<Capability>(implies, 1); }
  case Capability::RayTracingNV: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::PhysicalStorageBufferAddresses: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::CooperativeMatrixNV: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::FragmentShaderSampleInterlockEXT: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::FragmentShaderShadingRateInterlockEXT: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ShaderSMBuiltinsNV: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::FragmentShaderPixelInterlockEXT: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::DemoteToHelperInvocationEXT: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::IntegerFunctions2INTEL: {static const Capability implies[1] = {Capability::Shader}; return ArrayRef<Capability>(implies, 1); }
  case Capability::TessellationPointSize: {static const Capability implies[1] = {Capability::Tessellation}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GeometryPointSize: {static const Capability implies[1] = {Capability::Geometry}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageCubeArray: {static const Capability implies[1] = {Capability::SampledCubeArray}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ImageRect: {static const Capability implies[1] = {Capability::SampledRect}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GeometryStreams: {static const Capability implies[1] = {Capability::Geometry}; return ArrayRef<Capability>(implies, 1); }
  case Capability::MultiViewport: {static const Capability implies[1] = {Capability::Geometry}; return ArrayRef<Capability>(implies, 1); }
  case Capability::VariablePointers: {static const Capability implies[1] = {Capability::VariablePointersStorageBuffer}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SampleMaskOverrideCoverageNV: {static const Capability implies[1] = {Capability::SampleRateShading}; return ArrayRef<Capability>(implies, 1); }
  case Capability::GeometryShaderPassthroughNV: {static const Capability implies[1] = {Capability::Geometry}; return ArrayRef<Capability>(implies, 1); }
  case Capability::PerViewAttributesNV: {static const Capability implies[1] = {Capability::MultiView}; return ArrayRef<Capability>(implies, 1); }
  case Capability::InputAttachmentArrayDynamicIndexing: {static const Capability implies[1] = {Capability::InputAttachment}; return ArrayRef<Capability>(implies, 1); }
  case Capability::UniformBufferArrayNonUniformIndexing: {static const Capability implies[1] = {Capability::ShaderNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::SampledImageArrayNonUniformIndexing: {static const Capability implies[1] = {Capability::ShaderNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageBufferArrayNonUniformIndexing: {static const Capability implies[1] = {Capability::ShaderNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::StorageImageArrayNonUniformIndexing: {static const Capability implies[1] = {Capability::ShaderNonUniform}; return ArrayRef<Capability>(implies, 1); }
  case Capability::InputAttachmentArrayNonUniformIndexing: {static const Capability implies[2] = {Capability::InputAttachment, Capability::ShaderNonUniform}; return ArrayRef<Capability>(implies, 2); }
  case Capability::UniformTexelBufferArrayNonUniformIndexing: {static const Capability implies[2] = {Capability::SampledBuffer, Capability::ShaderNonUniform}; return ArrayRef<Capability>(implies, 2); }
  case Capability::StorageTexelBufferArrayNonUniformIndexing: {static const Capability implies[2] = {Capability::ImageBuffer, Capability::ShaderNonUniform}; return ArrayRef<Capability>(implies, 2); }
  case Capability::ShaderViewportIndexLayerEXT: {static const Capability implies[1] = {Capability::MultiViewport}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ShaderViewportMaskNV: {static const Capability implies[1] = {Capability::ShaderViewportIndexLayerEXT}; return ArrayRef<Capability>(implies, 1); }
  case Capability::ShaderStereoViewNV: {static const Capability implies[1] = {Capability::ShaderViewportMaskNV}; return ArrayRef<Capability>(implies, 1); }
  }
}
