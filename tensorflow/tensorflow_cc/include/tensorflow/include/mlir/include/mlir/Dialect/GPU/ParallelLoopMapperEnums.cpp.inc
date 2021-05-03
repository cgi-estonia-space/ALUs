/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyProcessor(Processor val) {
  switch (val) {
    case Processor::BlockX: return "BlockX";
    case Processor::BlockY: return "BlockY";
    case Processor::BlockZ: return "BlockZ";
    case Processor::ThreadX: return "ThreadX";
    case Processor::ThreadY: return "ThreadY";
    case Processor::ThreadZ: return "ThreadZ";
    case Processor::Sequential: return "Sequential";
  }
  return "";
}

::llvm::Optional<Processor> symbolizeProcessor(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Processor>>(str)
      .Case("BlockX", Processor::BlockX)
      .Case("BlockY", Processor::BlockY)
      .Case("BlockZ", Processor::BlockZ)
      .Case("ThreadX", Processor::ThreadX)
      .Case("ThreadY", Processor::ThreadY)
      .Case("ThreadZ", Processor::ThreadZ)
      .Case("Sequential", Processor::Sequential)
      .Default(::llvm::None);
}
::llvm::Optional<Processor> symbolizeProcessor(uint64_t value) {
  switch (value) {
  case 0: return Processor::BlockX;
  case 1: return Processor::BlockY;
  case 2: return Processor::BlockZ;
  case 3: return Processor::ThreadX;
  case 4: return Processor::ThreadY;
  case 5: return Processor::ThreadZ;
  case 6: return Processor::Sequential;
  default: return ::llvm::None;
  }
}

} // namespace gpu
} // namespace mlir

