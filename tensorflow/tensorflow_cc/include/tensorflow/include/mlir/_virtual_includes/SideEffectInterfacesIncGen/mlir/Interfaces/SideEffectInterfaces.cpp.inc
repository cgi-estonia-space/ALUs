/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

void mlir::MemoryEffectOpInterface::getEffects(SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffects(getOperation(), effects);
  }
void mlir::MemoryEffectOpInterface::getEffectsOnValue(Value value, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffectsOnValue(getOperation(), value, effects);
  }
void mlir::MemoryEffectOpInterface::getEffectsOnResource(SideEffects::Resource * resource, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffectsOnResource(getOperation(), resource, effects);
  }
