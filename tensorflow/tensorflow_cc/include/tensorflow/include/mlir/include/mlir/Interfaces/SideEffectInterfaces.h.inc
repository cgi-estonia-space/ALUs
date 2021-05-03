/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct MemoryEffectOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual void getEffects(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) const = 0;
    virtual void getEffectsOnValue(::mlir::Operation *tablegen_opaque_val, Value value, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) const = 0;
    virtual void getEffectsOnResource(::mlir::Operation *tablegen_opaque_val, SideEffects::Resource * resource, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    void getEffects(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEffects(effects);
    }
    void getEffectsOnValue(::mlir::Operation *tablegen_opaque_val, Value value, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) const final {
      (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEffects(effects);
          llvm::erase_if(effects, [&](auto &it) {
            return it.getValue() != value;
          });
    }
    void getEffectsOnResource(::mlir::Operation *tablegen_opaque_val, SideEffects::Resource * resource, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) const final {
      (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEffects(effects);
          llvm::erase_if(effects, [&](auto &it) {
            return it.getResource() != resource;
          });
    }
  };
};
} // end namespace detail
class MemoryEffectOpInterface : public ::mlir::OpInterface<MemoryEffectOpInterface, detail::MemoryEffectOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MemoryEffectOpInterface, detail::MemoryEffectOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct MemoryEffectOpInterfaceTrait : public ::mlir::OpInterface<MemoryEffectOpInterface, detail::MemoryEffectOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public MemoryEffectOpInterfaceTrait<ConcreteOp> {};
  void getEffects(SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects);
  void getEffectsOnValue(Value value, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects);
  void getEffectsOnResource(SideEffects::Resource * resource, SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects);

    /// Collect all of the effect instances that correspond to the given
    /// `Effect` and place them in 'effects'.
    template <typename Effect> void getEffects(
      SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                                              ::mlir::MemoryEffects::Effect>> &effects) {
      getEffects(effects);
      llvm::erase_if(effects, [&](auto &it) {
        return !llvm::isa<Effect>(it.getEffect());
      });
    }

    /// Returns true if this operation exhibits the given effect.
    template <typename Effect> bool hasEffect() {
      SmallVector<SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>, 4> effects;
      getEffects(effects);
      return llvm::any_of(effects, [](const auto &it) {
        return llvm::isa<Effect>(it.getEffect());
      });
    }

    /// Returns true if this operation only has the given effect.
    template <typename Effect> bool onlyHasEffect() {
      SmallVector<SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>, 4> effects;
      getEffects(effects);
      return !effects.empty() && llvm::all_of(effects, [](const auto &it) {
        return isa<Effect>(it.getEffect());
      });
    }

    /// Returns true if this operation has no effects.
    bool hasNoEffect() {
      SmallVector<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>, 4> effects;
      getEffects(effects);
      return effects.empty();
    }

    /// Returns true if the given operation has no effects for this interface.
    static bool hasNoEffect(Operation *op) {
      if (auto interface = dyn_cast<MemoryEffectOpInterface>(op))
        return interface.hasNoEffect();
      return op->hasTrait<::mlir::OpTrait::HasRecursiveSideEffects>();
    }
  
};
} // namespace mlir
