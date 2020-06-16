/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the operations used in the XLA dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_IR_HLO_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_IR_HLO_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Dialect.h"  // TF:llvm-project
#include "mlir/IR/DialectImplementation.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/Support/Functional.h"  // TF:llvm-project

namespace mlir {
class OpBuilder;

#include "tensorflow/compiler/mlir/xla/ir/hlo_structs.h.inc"

namespace xla_hlo {

class XlaHloDialect : public Dialect {
 public:
  explicit XlaHloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "xla_hlo"; }

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

namespace HLOTypes {
enum Kind {
  Token = Type::FIRST_XLA_HLO_TYPE,
};
}  // namespace HLOTypes

class TokenType : public Type::TypeBase<TokenType, Type> {
 public:
  using Base::Base;

  static TokenType get(MLIRContext *context) {
    return Base::get(context, HLOTypes::Token);
  }

  // Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == HLOTypes::Token; }
};

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h.inc"

}  // end namespace xla_hlo
}  // end namespace mlir

#endif  //  TENSORFLOW_COMPILER_MLIR_XLA_IR_HLO_OPS_H_
