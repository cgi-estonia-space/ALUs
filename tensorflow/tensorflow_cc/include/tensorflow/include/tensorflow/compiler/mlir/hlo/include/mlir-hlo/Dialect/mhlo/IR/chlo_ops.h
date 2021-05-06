/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_CHLO_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_CHLO_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace chlo {

class HloClientDialect : public Dialect {
  void initialize();

 public:
  explicit HloClientDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<HloClientDialect>()) {
    initialize();
  }
  static StringRef getDialectNamespace() { return "chlo"; }
};

}  // namespace chlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h.inc"

namespace mlir {
namespace chlo {

template <typename T>
static Value getConstantLike(OpBuilder& b, Location loc, T constant,
                             Value val) {
  Type ty = getElementTypeOrSelf(val.getType());

  auto getAttr = [&]() -> Attribute {
    if (ty.isa<IntegerType>()) return b.getIntegerAttr(ty, constant);
    if (ty.isa<FloatType>()) return b.getFloatAttr(ty, constant);
    llvm_unreachable("unhandled element type");
  };
  return b.create<ConstantLikeOp>(loc, getAttr(), val);
}

}  // namespace chlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_CHLO_OPS_H_
