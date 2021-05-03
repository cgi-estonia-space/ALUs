/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct RegionKindInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual RegionKind getRegionKind(unsigned index) const = 0;
    virtual bool hasSSADominance(unsigned index) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    RegionKind getRegionKind(unsigned index) const final {
      return ConcreteOp::getRegionKind(index);
    }
    bool hasSSADominance(unsigned index) const final {
      return getRegionKind(index) == RegionKind::SSACFG;
    }
  };
};
} // end namespace detail
class RegionKindInterface : public ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct RegionKindInterfaceTrait : public ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public RegionKindInterfaceTrait<ConcreteOp> {};
  RegionKind getRegionKind(unsigned index);
  bool hasSSADominance(unsigned index);
};
} // namespace mlir
