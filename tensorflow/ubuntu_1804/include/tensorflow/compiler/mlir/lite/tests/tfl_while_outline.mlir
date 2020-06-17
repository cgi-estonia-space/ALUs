// Test to verify loop outlining.

// RUN: tf-opt --split-input-file --tfl-while-loop-outline %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: func @while
func @while() -> tensor<1xf32>
    attributes {tf.entry_function = {outputs = "result"}} {
  %cst = constant dense<1> : tensor<i32> loc("dec")
  %arg0 = constant dense<5> : tensor<i32> loc("N")
  %arg1 = constant dense<3.0> : tensor<1xf32> loc("val")
  %0:2 = "tfl.while"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      // CHECK: call @WhileOp_cond
      %cst_0 = constant dense<0> : tensor<i32>
      %1 = "tfl.greater"(%arg2, %cst_0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      // CHECK: call @WhileOp_body
      %1 = "tfl.sub"(%arg2, %cst) {fused_activation_function = "NONE"} :
        (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = tfl.add %arg3, %arg3 {fused_activation_function = "NONE"} : tensor<*xf32>
      "tfl.yield"(%1, %2) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) loc("WhileOp")
  return %0#1 : tensor<1xf32>
}
// CHECK-LABEL: func @WhileOp_cond(
// CHECK: tfl.greater
// CHECK-LABEL: func @WhileOp_body(
// CHECK: tfl.sub
// CHECK: tfl.add

// -----

func @rnn(%arg0: tensor<4x4x3xf32> {tf.device = "/device:CPU:0"}) -> tensor<4x?x2xf32> attributes {tf.entry_function = {inputs = "Placeholder", outputs = "rnn/transpose_1"}} {
  %cst = constant dense<0.000000e+00> : tensor<4x2xf32>
  %cst_0 = constant dense<0.000000e+00> : tensor<8xf32>
  %cst_1 = constant dense<[1, 0, 2]> : tensor<3xi32>
  %cst_2 = constant dense<0.000000e+00> : tensor<4x4x2xf32>
  %cst_3 = constant dense<4> : tensor<i32>
  %cst_4 = constant dense<1.000000e+00> : tensor<f32>
  %cst_5 = constant dense<1> : tensor<i32>
  %cst_6 = constant dense<0> : tensor<1xi32>
  %cst_7 = constant dense<0> : tensor<i32>
  %cst_8 = constant dense<-1> : tensor<1xi32>
  %cst_9 = constant dense<-1> : tensor<i32>
  %cst_10 = constant dense<2.1> : tensor<8x5xf32>
  %cst_11 = constant dense<2> : tensor<1xi32>
  %cst_12 = constant dense<1> : tensor<1xi32>
  %0 = "tfl.transpose"(%arg0, %cst_1) : (tensor<4x4x3xf32>, tensor<3xi32>) -> tensor<4x4x3xf32>
  %1:6 = "tfl.while"(%cst_7, %cst_7, %cst_2, %cst, %cst, %0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<*xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<4x2xf32>, %arg6: tensor<*xf32>):  // no predecessors
    %5 = "tfl.less"(%arg2, %cst_3) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = "tfl.less"(%arg1, %cst_3) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = tfl.logical_and %6, %5 : tensor<i1>
    "tfl.yield"(%7) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<*xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<4x2xf32>, %arg6: tensor<*xf32>):  // no predecessors
    %5 = tfl.add %arg2, %cst_5 {fused_activation_function = "NONE"} : tensor<i32>
    %6 = tfl.add %arg1, %cst_5 {fused_activation_function = "NONE"} : tensor<i32>
    %7 = "tfl.gather"(%0, %arg2) {axis = 0 : i32} : (tensor<4x4x3xf32>, tensor<i32>) -> tensor<4x3xf32>
    %8 = "tfl.concatenation"(%7, %arg5) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<4x3xf32>, tensor<4x2xf32>) -> tensor<4x5xf32>
    %9 = "tfl.fully_connected"(%8, %cst_10, %cst_0) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x5xf32>, tensor<8x5xf32>, tensor<8xf32>) -> tensor<4x8xf32>
    %10:4 = "tfl.split"(%cst_5, %9) {num_splits = 4 : i32} : (tensor<i32>, tensor<4x8xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>)
    %11 = "tfl.add"(%10#2, %cst_4) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
    %12 = "tfl.logistic"(%11) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %13 = tfl.mul %arg4, %12 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %14 = "tfl.relu"(%10#1) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %15 = "tfl.logistic"(%10#0) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %16 = tfl.mul %15, %14 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %17 = tfl.add %13, %16 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %18 = "tfl.relu"(%17) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %19 = "tfl.logistic"(%10#3) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %20 = tfl.mul %18, %19 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %21 = "tfl.fill"(%cst_11, %cst_7) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
    %22 = "tfl.concatenation"(%cst_6, %21) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %23 = "tfl.reshape"(%arg2, %cst_12) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %24 = "tfl.fill"(%cst_11, %cst_9) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
    %25 = "tfl.concatenation"(%23, %24) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %26 = "tfl.slice"(%arg3, %22, %25) : (tensor<*xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
    %27 = "tfl.reshape"(%5, %cst_12) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %28 = "tfl.concatenation"(%27, %21) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %29 = "tfl.concatenation"(%cst_8, %24) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %30 = "tfl.slice"(%arg3, %28, %29) : (tensor<*xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
    %31 = "tfl.expand_dims"(%20, %cst_7) : (tensor<4x2xf32>, tensor<i32>) -> tensor<*xf32>
    %32 = "tfl.concatenation"(%26, %31, %30) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tfl.yield"(%6, %5, %32, %17, %20, %0) : (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x4x3xf32>) -> ()
  }) {is_stateless = true} : (tensor<i32>, tensor<i32>, tensor<4x4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x4x3xf32>) -> (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>)
  %2 = "tfl.shape"(%1#2) : (tensor<*xf32>) -> tensor<?xi32>
  %3 = "tfl.reshape"(%1#2, %2) : (tensor<*xf32>, tensor<?xi32>) -> tensor<?x4x2xf32>
  %4 = "tfl.transpose"(%3, %cst_1) : (tensor<?x4x2xf32>, tensor<3xi32>) -> tensor<4x?x2xf32>
  return %4 : tensor<4x?x2xf32>
}

// CHECK-LABEL:   func @rnn(
// CHECK:           tfl.while
// CHECK:             tfl.yield
// CHECK-SAME:  (tensor<i1>) -> ()
// CHECK:             [[VAL_41:%.*]]:18 =
// CHECK: call @tfl.while_body
// CHECK:             tfl.yield
// CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>, tensor<i32>, tensor<i32>, tensor<4x4x3xf32>, tensor<8x5xf32>, tensor<8xf32>, tensor<f32>, tensor<1xi32>, tensor<i32>, tensor<1xi32>, tensor<1xi32>, tensor<i32>, tensor<1xi32>) -> ()

// CHECK-LABEL:   func @tfl.while_cond(
// CHECK-SAME:                         [[VAL_56:%.*]]: tensor<i32>, [[VAL_57:%.*]]: tensor<i32>, [[VAL_58:%.*]]: tensor<*xf32>, [[VAL_59:%.*]]: tensor<4x2xf32>, [[VAL_60:%.*]]: tensor<4x2xf32>, [[VAL_61:%.*]]: tensor<*xf32>, [[VAL_62:%.*]]: tensor<i32>, [[VAL_63:%.*]]: tensor<i32>, [[VAL_64:%.*]]: tensor<4x4x3xf32>, [[VAL_65:%.*]]: tensor<8x5xf32>, [[VAL_66:%.*]]: tensor<8xf32>, [[VAL_67:%.*]]: tensor<f32>, [[VAL_68:%.*]]: tensor<1xi32>, [[VAL_69:%.*]]: tensor<i32>, [[VAL_70:%.*]]: tensor<1xi32>, [[VAL_71:%.*]]: tensor<1xi32>, [[VAL_72:%.*]]: tensor<i32>, [[VAL_73:%.*]]: tensor<1xi32>) -> tensor<i1> attributes {sym_visibility = "private"} {
// CHECK:           return
// CHECK-SAME:        tensor<i1>
// CHECK:         }

// CHECK-LABEL:   func @tfl.while_body(
// CHECK-SAME:                         [[VAL_77:%.*]]: tensor<i32>, [[VAL_78:%.*]]: tensor<i32>, [[VAL_79:%.*]]: tensor<*xf32>, [[VAL_80:%.*]]: tensor<4x2xf32>, [[VAL_81:%.*]]: tensor<4x2xf32>, [[VAL_82:%.*]]: tensor<*xf32>, [[VAL_83:%.*]]: tensor<i32>, [[VAL_84:%.*]]: tensor<i32>, [[VAL_85:%.*]]: tensor<4x4x3xf32>, [[VAL_86:%.*]]: tensor<8x5xf32>, [[VAL_87:%.*]]: tensor<8xf32>, [[VAL_88:%.*]]: tensor<f32>, [[VAL_89:%.*]]: tensor<1xi32>, [[VAL_90:%.*]]: tensor<i32>, [[VAL_91:%.*]]: tensor<1xi32>, [[VAL_92:%.*]]: tensor<1xi32>, [[VAL_93:%.*]]: tensor<i32>, [[VAL_94:%.*]]: tensor<1xi32>) -> (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>, tensor<i32>, tensor<i32>, tensor<4x4x3xf32>, tensor<8x5xf32>, tensor<8xf32>, tensor<f32>, tensor<1xi32>, tensor<i32>, tensor<1xi32>, tensor<1xi32>, tensor<i32>, tensor<1xi32>) attributes {sym_visibility = "private"} {
// CHECK:           [[VAL_123:%.*]] = "tfl.cast"
// CHECK:           return
// CHECK-SAME:        [[VAL_123]], [[VAL_83]], [[VAL_84]], [[VAL_85]], [[VAL_86]], [[VAL_87]], [[VAL_88]], [[VAL_89]], [[VAL_90]], [[VAL_91]], [[VAL_92]], [[VAL_93]], [[VAL_94]] : tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>, tensor<i32>, tensor<i32>, tensor<4x4x3xf32>, tensor<8x5xf32>, tensor<8xf32>, tensor<f32>, tensor<1xi32>, tensor<i32>, tensor<1xi32>, tensor<1xi32>, tensor<i32>, tensor<1xi32>
// CHECK:         }
// CHECK:       }

