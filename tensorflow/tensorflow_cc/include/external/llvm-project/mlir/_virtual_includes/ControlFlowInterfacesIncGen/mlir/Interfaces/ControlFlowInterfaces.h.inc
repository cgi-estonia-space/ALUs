/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct BranchOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Optional<MutableOperandRange> getMutableSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const = 0;
    virtual Optional<OperandRange> getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const = 0;
    virtual Optional<BlockArgument> getSuccessorBlockArgument(::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) const = 0;
    virtual Block *getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ArrayRef<Attribute> operands) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Optional<MutableOperandRange> getMutableSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(index);
    }
    Optional<OperandRange> getSuccessorOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(index);
    }
    Optional<BlockArgument> getSuccessorBlockArgument(::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) const final {
      Operation *opaqueOp = (llvm::cast<ConcreteOp>(tablegen_opaque_val));
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (Optional<BlockArgument> arg = detail::getBranchSuccessorArgument(
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return llvm::None;
    }
    Block *getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ArrayRef<Attribute> operands) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorForOperands(operands);
    }
  };
};
} // end namespace detail
class BranchOpInterface : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct BranchOpInterfaceTrait : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Optional<OperandRange> getSuccessorOperands(unsigned index) {
      ConcreteOp *op = static_cast<ConcreteOp *>(this);
        auto operands = op->getMutableSuccessorOperands(index);
        return operands ? Optional<OperandRange>(*operands) : llvm::None;
    }
    Block *getSuccessorForOperands(ArrayRef<Attribute> operands) {
      return nullptr;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      auto concreteOp = cast<ConcreteOpType>(op);
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
      Optional<OperandRange> operands = concreteOp.getSuccessorOperands(i);
      if (failed(detail::verifyBranchSuccessorOperands(op, i, operands)))
        return failure();
    }
    return success();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public BranchOpInterfaceTrait<ConcreteOp> {};
  Optional<MutableOperandRange> getMutableSuccessorOperands(unsigned index);
  Optional<OperandRange> getSuccessorOperands(unsigned index);
  Optional<BlockArgument> getSuccessorBlockArgument(unsigned operandIndex);
  Block *getSuccessorForOperands(ArrayRef<Attribute> operands);
};
} // namespace mlir
namespace mlir {
namespace detail {
struct RegionBranchOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual OperandRange getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const = 0;
    virtual void getSuccessorRegions(::mlir::Operation *tablegen_opaque_val, Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    OperandRange getSuccessorEntryOperands(::mlir::Operation *tablegen_opaque_val, unsigned index) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorEntryOperands(index);
    }
    void getSuccessorRegions(::mlir::Operation *tablegen_opaque_val, Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorRegions(index, operands, regions);
    }
  };
};
} // end namespace detail
class RegionBranchOpInterface : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct RegionBranchOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    OperandRange getSuccessorEntryOperands(unsigned index) {
      auto operandEnd = this->getOperation()->operand_end();
        return OperandRange(operandEnd, operandEnd);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      static_assert(!ConcreteOpType::template hasTrait<OpTrait::ZeroRegion>(),
                  "expected operation to have non-zero regions");
    return success();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public RegionBranchOpInterfaceTrait<ConcreteOp> {};
  OperandRange getSuccessorEntryOperands(unsigned index);
  void getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions);

    /// Verify types along control flow edges described by this interface.
    static LogicalResult verifyTypes(Operation *op) {
      return detail::verifyTypesAlongControlFlowEdges(op);
    }
  
};
} // namespace mlir
