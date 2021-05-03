/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace linalg {
namespace detail {
struct LinalgOpInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual unsigned getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual void getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) const = 0;
    virtual unsigned getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual void getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) const = 0;
    virtual unsigned getNumWindowLoops(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual void getWindowDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) const = 0;
    virtual unsigned getNumLoops(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual unsigned getNumInputs(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual unsigned getNumInitTensors(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual unsigned getNumOutputs(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Value getInput(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual llvm::Optional<unsigned> getIndexOfInput(::mlir::Operation *tablegen_opaque_val, Value value) const = 0;
    virtual ShapedType getInputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual Operation::operand_range getInputs(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual SmallVector<RankedTensorType, 4> getInputTensorTypes(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Value getOutputBuffer(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual llvm::Optional<unsigned> getIndexOfOutputBuffer(::mlir::Operation *tablegen_opaque_val, Value value) const = 0;
    virtual MemRefType getOutputBufferType(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual ShapedType getOutputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual SmallVector<RankedTensorType, 4> getOutputTensorTypes(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Operation::operand_range getOutputBuffers(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Value getBuffer(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual unsigned getNumInputsAndOutputs(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual unsigned getNumInputsAndOutputBuffers(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Operation::operand_range getInputsAndOutputBuffers(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Operation::operand_range getInitTensors(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Value getInitTensor(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual unsigned getNumShapedOperands(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Operation::operand_range getShapedOperands(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual ShapedType getShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual SmallVector<ShapedType, 4> getInputOutputShapedTypes(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Optional<unsigned> getIndexOfInputAndOutputBuffer(::mlir::Operation *tablegen_opaque_val, Value value) const = 0;
    virtual ArrayAttr iterator_types(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual ArrayAttr indexing_maps(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual SmallVector<AffineMap, 4> getIndexingMaps(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual AffineMap getIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual AffineMap getInputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual AffineMap getOutputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) const = 0;
    virtual bool hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual std::string getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Operation *create(OpBuilder & builder, Location loc, TypeRange resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) const = 0;
    virtual Operation *clone(::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    unsigned getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumParallelLoops();
    }
    void getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getParallelDims(res);
    }
    unsigned getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumReductionLoops();
    }
    void getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReductionDims(res);
    }
    unsigned getNumWindowLoops(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumWindowLoops();
    }
    void getWindowDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<AffineExpr> & res) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getWindowDims(res);
    }
    unsigned getNumLoops(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumLoops();
    }
    bool hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasSingleReductionLoop();
    }
    unsigned getNumInputs(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs();
    }
    unsigned getNumInitTensors(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInitTensors();
    }
    unsigned getNumOutputs(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOutputs();
    }
    Value getInput(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInput(i);
    }
    llvm::Optional<unsigned> getIndexOfInput(::mlir::Operation *tablegen_opaque_val, Value value) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfInput(value);
    }
    ShapedType getInputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputShapedType(i);
    }
    Operation::operand_range getInputs(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputs();
    }
    SmallVector<RankedTensorType, 4> getInputTensorTypes(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputTensorTypes();
    }
    Value getOutputBuffer(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBuffer(i);
    }
    llvm::Optional<unsigned> getIndexOfOutputBuffer(::mlir::Operation *tablegen_opaque_val, Value value) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfOutputBuffer(value);
    }
    MemRefType getOutputBufferType(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBufferType(i);
    }
    ShapedType getOutputShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputShapedType(i);
    }
    SmallVector<RankedTensorType, 4> getOutputTensorTypes(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensorTypes();
    }
    Operation::operand_range getOutputBuffers(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBuffers();
    }
    Value getBuffer(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBuffer(i);
    }
    unsigned getNumInputsAndOutputs(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputsAndOutputs();
    }
    unsigned getNumInputsAndOutputBuffers(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputsAndOutputBuffers();
    }
    Operation::operand_range getInputsAndOutputBuffers(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputsAndOutputBuffers();
    }
    Operation::operand_range getInitTensors(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInitTensors();
    }
    Value getInitTensor(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInitTensor(i);
    }
    unsigned getNumShapedOperands(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumShapedOperands();
    }
    Operation::operand_range getShapedOperands(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedOperands();
    }
    ShapedType getShapedType(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedType(i);
    }
    SmallVector<ShapedType, 4> getInputOutputShapedTypes(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputOutputShapedTypes();
    }
    Optional<unsigned> getIndexOfInputAndOutputBuffer(::mlir::Operation *tablegen_opaque_val, Value value) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfInputAndOutputBuffer(value);
    }
    ArrayAttr iterator_types(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types();
    }
    ArrayAttr indexing_maps(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps();
    }
    SmallVector<AffineMap, 4> getIndexingMaps(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps();
    }
    AffineMap getIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMap(i);
    }
    AffineMap getInputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputIndexingMap(i);
    }
    AffineMap getOutputIndexingMap(::mlir::Operation *tablegen_opaque_val, unsigned i) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputIndexingMap(i);
    }
    bool hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasBufferSemantics();
    }
    bool hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasTensorSemantics();
    }
    std::string getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLibraryCallName();
    }
    Operation *create(OpBuilder & builder, Location loc, TypeRange resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) const final {
      return builder.create<ConcreteOp>(
          loc, resultTypes, operands, attributes);
    }
    Operation *clone(::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) const final {
      BlockAndValueMapping map;
        unsigned numRegions = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getNumRegions();
        Operation *res = create(b, loc, resultTypes, operands, (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAttrs());
        assert(res->getNumRegions() == numRegions && "inconsistent # regions");
        for (unsigned ridx = 0; ridx < numRegions; ++ridx)
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getRegion(ridx).cloneInto(
            &res->getRegion(ridx), map);
        return res;
    }
  };
};
} // end namespace detail
class LinalgOp : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct LinalgOpTrait : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::Trait<ConcreteOp> {
    unsigned getNumParallelLoops() {
      return getNumIterators(getParallelIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getParallelDims(SmallVectorImpl<AffineExpr> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getParallelIteratorTypeName(), res);
    }
    unsigned getNumReductionLoops() {
      return getNumIterators(getReductionIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getReductionDims(SmallVectorImpl<AffineExpr> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getReductionIteratorTypeName(), res);
    }
    unsigned getNumWindowLoops() {
      return getNumIterators(getWindowIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getWindowDims(SmallVectorImpl<AffineExpr> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)).getOperation(), getWindowIteratorTypeName(), res);
    }
    unsigned getNumLoops() {
      return getNumIterators((*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    bool hasSingleReductionLoop() {
      auto iters = (*static_cast<ConcreteOp *>(this)).iterator_types();
        return iters.size() == 1 &&
               getNumIterators(getReductionIteratorTypeName(), iters) == 1;
    }
    Value getInput(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumInputs());
        return this->getOperation()->getOperand(i);
    }
    llvm::Optional<unsigned> getIndexOfInput(Value value) {
      auto it = llvm::find(getInputs(), value);
        if (it != getInputs().end())
          return it - getInputs().begin();
        return llvm::None;
    }
    ShapedType getInputShapedType(unsigned i) {
      return getInput(i).getType().template cast<ShapedType>();
    }
    Operation::operand_range getInputs() {
      auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + (*static_cast<ConcreteOp *>(this)).getNumInputs()};
    }
    SmallVector<RankedTensorType, 4> getInputTensorTypes() {
      SmallVector<RankedTensorType, 4> res;
        for (Type type : getInputs().getTypes())
          if (auto t = type.template dyn_cast<RankedTensorType>())
            res.push_back(t);
        return res;
    }
    Value getOutputBuffer(unsigned i) {
      // Output buffers are passed as output buffer operands (side-effecting).
        // Output tensors are results.
        // The union of the 2 are all the outputs and we want to ensure i does
        // not overflow the buffer operands.
        assert(i + this->getOperation()->getNumResults() < (*static_cast<ConcreteOp *>(this)).getNumOutputs()
               && "overflowing output buffer index");
        return this->getOperation()->getOperand((*static_cast<ConcreteOp *>(this)).getNumInputs() + i);
    }
    llvm::Optional<unsigned> getIndexOfOutputBuffer(Value value) {
      auto it = llvm::find(getOutputBuffers(), value);
        if (it != getOutputBuffers().end())
          return it - getOutputBuffers().begin();
        return llvm::None;
    }
    MemRefType getOutputBufferType(unsigned i) {
      return getOutputBuffer(i).getType().template cast<MemRefType>();
    }
    ShapedType getOutputShapedType(unsigned i) {
      return getShapedType(i + (*static_cast<ConcreteOp *>(this)).getNumInputs());
    }
    SmallVector<RankedTensorType, 4> getOutputTensorTypes() {
      SmallVector<RankedTensorType, 4> res;
        for (Type type : this->getOperation()->getResults().getTypes())
          res.push_back(type.template cast<RankedTensorType>());
        return res;
    }
    Operation::operand_range getOutputBuffers() {
      auto range = this->getOperation()->getOperands();
        return {range.begin() + (*static_cast<ConcreteOp *>(this)).getNumInputs(),
                range.begin() + getNumInputsAndOutputBuffers()};
    }
    Value getBuffer(unsigned i) {
      assert(i < getNumInputsAndOutputBuffers() && "overflowing buffers index");
        return this->getOperation()->getOperand(i);
    }
    unsigned getNumInputsAndOutputs() {
      return (*static_cast<ConcreteOp *>(this)).getNumInputs() + (*static_cast<ConcreteOp *>(this)).getNumOutputs();
    }
    unsigned getNumInputsAndOutputBuffers() {
      return (*static_cast<ConcreteOp *>(this)).getNumInputs() + (*static_cast<ConcreteOp *>(this)).getNumOutputs() -
          this->getOperation()->getNumResults();
    }
    Operation::operand_range getInputsAndOutputBuffers() {
      auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + getNumInputsAndOutputBuffers()};
    }
    Operation::operand_range getInitTensors() {
      auto range = this->getOperation()->getOperands();
        return {range.begin() + getNumInputsAndOutputBuffers(),
                range.begin() + getNumInputsAndOutputs()};
    }
    Value getInitTensor(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumInitTensors() && "overflowing init tensor index");
        return getInitTensors()[i];
    }
    unsigned getNumShapedOperands() {
      return getNumInputsAndOutputBuffers() + (*static_cast<ConcreteOp *>(this)).getNumInitTensors();
    }
    Operation::operand_range getShapedOperands() {
      auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + getNumShapedOperands()};
    }
    ShapedType getShapedType(unsigned i) {
      if (i < (*static_cast<ConcreteOp *>(this)).getNumInputs())
          return getInputShapedType(i);
        if (i < getNumInputsAndOutputBuffers())
          return getOutputBufferType(i - (*static_cast<ConcreteOp *>(this)).getNumInputs());
        return this->getOperation()->getResult(
          i - getNumInputsAndOutputBuffers()).
          getType().template cast<ShapedType>();
    }
    SmallVector<ShapedType, 4> getInputOutputShapedTypes() {
      SmallVector<Type, 4> inputOutputTypes(
            this->getOperation()->operand_type_begin(),
            this->getOperation()->operand_type_end());
        inputOutputTypes.append(this->getOperation()->result_type_begin(),
                                this->getOperation()->result_type_end());
        return llvm::to_vector<4>(
            llvm::map_range(inputOutputTypes, [](Type type) -> ShapedType {
              return type.cast<ShapedType>();
            }));
    }
    Optional<unsigned> getIndexOfInputAndOutputBuffer(Value value) {
      Optional<unsigned> inputIndex = getIndexOfInput(value);
        if (inputIndex.hasValue()) return inputIndex.getValue();
        Optional<unsigned> outputIndex = getIndexOfOutputBuffer(value);
        if (outputIndex.hasValue()) {
          return (*static_cast<ConcreteOp *>(this)).getNumInputs() + outputIndex.getValue();
        }
        return llvm::None;
    }
    ArrayAttr iterator_types() {
      return (*static_cast<ConcreteOp *>(this)).iterator_types();
    }
    SmallVector<AffineMap, 4> getIndexingMaps() {
      return llvm::to_vector<4>(
          (*static_cast<ConcreteOp *>(this)).indexing_maps().template getAsValueRange<AffineMapAttr>());
    }
    AffineMap getIndexingMap(unsigned i) {
      assert(i < getNumInputsAndOutputs());
        return getIndexingMaps()[i];
    }
    AffineMap getInputIndexingMap(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumInputs());
        return getIndexingMaps()[i];
    }
    AffineMap getOutputIndexingMap(unsigned i) {
      assert(i < (*static_cast<ConcreteOp *>(this)).getNumOutputs());
        return getIndexingMaps()[i + (*static_cast<ConcreteOp *>(this)).getNumInputs()];
    }
    bool hasBufferSemantics() {
      return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(getInputs(),
                       [](Value v) { return v.getType().isa<MemRefType>(); });
    }
    bool hasTensorSemantics() {
      auto isTensorType = [](Value v) {
          return v.getType().isa<RankedTensorType>();
        };
        return llvm::all_of(getInputs(), isTensorType) &&
               llvm::all_of(this->getOperation()->getResults(), isTensorType);
    }
    std::string getLibraryCallName() {
      return (*static_cast<ConcreteOp *>(this)).getLibraryCallName();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public LinalgOpTrait<ConcreteOp> {};
  unsigned getNumParallelLoops();
  void getParallelDims(SmallVectorImpl<AffineExpr> & res);
  unsigned getNumReductionLoops();
  void getReductionDims(SmallVectorImpl<AffineExpr> & res);
  unsigned getNumWindowLoops();
  void getWindowDims(SmallVectorImpl<AffineExpr> & res);
  unsigned getNumLoops();
  bool hasSingleReductionLoop();
  unsigned getNumInputs();
  unsigned getNumInitTensors();
  unsigned getNumOutputs();
  Value getInput(unsigned i);
  llvm::Optional<unsigned> getIndexOfInput(Value value);
  ShapedType getInputShapedType(unsigned i);
  Operation::operand_range getInputs();
  SmallVector<RankedTensorType, 4> getInputTensorTypes();
  Value getOutputBuffer(unsigned i);
  llvm::Optional<unsigned> getIndexOfOutputBuffer(Value value);
  MemRefType getOutputBufferType(unsigned i);
  ShapedType getOutputShapedType(unsigned i);
  SmallVector<RankedTensorType, 4> getOutputTensorTypes();
  Operation::operand_range getOutputBuffers();
  Value getBuffer(unsigned i);
  unsigned getNumInputsAndOutputs();
  unsigned getNumInputsAndOutputBuffers();
  Operation::operand_range getInputsAndOutputBuffers();
  Operation::operand_range getInitTensors();
  Value getInitTensor(unsigned i);
  unsigned getNumShapedOperands();
  Operation::operand_range getShapedOperands();
  ShapedType getShapedType(unsigned i);
  SmallVector<ShapedType, 4> getInputOutputShapedTypes();
  Optional<unsigned> getIndexOfInputAndOutputBuffer(Value value);
  ArrayAttr iterator_types();
  ArrayAttr indexing_maps();
  SmallVector<AffineMap, 4> getIndexingMaps();
  AffineMap getIndexingMap(unsigned i);
  AffineMap getInputIndexingMap(unsigned i);
  AffineMap getOutputIndexingMap(unsigned i);
  bool hasBufferSemantics();
  bool hasTensorSemantics();
  std::string getLibraryCallName();
  Operation *create(OpBuilder & builder, Location loc, TypeRange resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  Operation *clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);

    /// Returns all the operands past the inputs, output_buffers and
    /// init_tensors operands. Asserts that these operands are value types to
    /// allow transformations like tiling to just use the values when cloning
    /// `linalgOp`.
    SmallVector<Value, 4> getAssumedNonShapedOperands() {
      unsigned numShapedOperands = getNumInputsAndOutputs();
      unsigned nExtraOperands =
        getOperation()->getNumOperands() - numShapedOperands;
      SmallVector<Value, 4> res;
      res.reserve(nExtraOperands);
      for (unsigned i = 0; i < nExtraOperands; ++i) {
        res.push_back(getOperation()->getOperand(numShapedOperands + i));
        assert((res.back().getType().isSignlessIntOrIndexOrFloat()
                || res.back().getType().isa<VectorType>()) &&
               "expected scalar or vector type");
      }
      return res;
    }
    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }
    void setNumInitTensors(unsigned num) { setOperandSegmentAt(2, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = getOperation()->getAttr("operand_segment_sizes")
        .cast<DenseIntElementsAttr>();
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(32, getContext()),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  
};
} // namespace linalg
} // namespace mlir
