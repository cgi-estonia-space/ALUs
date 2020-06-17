// RUN: tf-opt %s -split-input-file -tf-tpu-variable-runtime-reformatting| FileCheck %s --dump-input=fail

// Tests that the pass can correctly transform a training loop with 2 replicas.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: func @main
  func @main(%arg0: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
             %arg1: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
             %arg2: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
             %arg3: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"}) {

    %0 = "tf.Const"() {value = dense<100> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[STATE0:.*]] = "tf.VarHandleOp"()
    // CHECK-SAME: device = "/device:TPU:0"
    // CHECK: %[[STATE1:.*]] = "tf.VarHandleOp"()
    // CHECK-SAME: device = "/device:TPU:1"
    // CHECK: %[[WHILE:.*]]:7 = "tf.While"(
    // CHECK-SAME: %[[STATE0]], %[[STATE1]])
    %1:5 = "tf.While"(%0, %arg0, %arg1, %arg2, %arg3)
               {T = ["tfdtype$DT_INT32", "tfdtype$DT_RESOURCE",
                 "tfdtype$DT_RESOURCE", "tfdtype$DT_RESOURCE",
                 "tfdtype$DT_RESOURCE"], body = @while_body_7560,
                cond = @while_cond_7550, device = "", is_stateless = false,
                output_shapes = ["tfshape$", "tfshape$", "tfshape$", "tfshape$", "tfshape$"]}
         : (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>,
            tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>)
         -> (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>,
             tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>)
    // CHECK: %[[DEFAULT:.*]] = "tf.Const"()
    // CHECK:  tf_device.replicate
    // CHECK-SAME: as %[[V0:.*]]: tensor<*x!tf.resource<tensor<f32>>>,
    // CHECK-SAME: as %[[V1:.*]]: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>,
    // CHECK-SAME: [%[[STATE0]], %[[STATE1]]] as %[[STATE:.*]]: tensor<!tf.resource<tensor<2x!tf.string>>>
    // CHECK: "tf.TPUReshardVariables"(%[[V0]], %[[V1]], %[[DEFAULT]], %[[STATE]])
    return
  }
  // CHECK: func @while_body_7560
  func @while_body_7560(%arg0: tensor<i32>,
                        %arg1: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
                        %arg2: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
                        %arg3: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
                        %arg4: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"})
        -> (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>,
            tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>) {
    // CHECK-SAME: (%[[ITER:.*]]: tensor<i32>,
    // CHECK-SAME: %[[BODY_ARG1:.*]]: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
    // CHECK-SAME: %[[BODY_ARG2:.*]]: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
    // CHECK-SAME: %[[BODY_ARG3:.*]]: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
    // CHECK-SAME: %[[BODY_ARG4:.*]]: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"},
    // CHECK-SAME: %[[STATE_ARG0:.*]]: tensor<!tf.resource<tensor<2x!tf.string>>> {tf.device = "/device:TPU:0"},
    // CHECK-SAME: %[[STATE_ARG1:.*]]: tensor<!tf.resource<tensor<2x!tf.string>>> {tf.device = "/device:TPU:1"})
    %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.AddV2"(%arg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK: %[[COMPILE:.*]]:2 = "tf._TPUCompileMlir"()
    %2:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64, device = "/device:CPU:0",
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    "tf.TPUCompileSucceededAssert"(%2#0) : (tensor<!tf.string>) -> ()
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[BODY_ARG1]], %[[BODY_ARG2]]] as %[[R0:.*]]: tensor<*x!tf.resource<tensor<f32>>>,
    // CHECK-SAME: [%[[BODY_ARG3]], %[[BODY_ARG4]]] as %[[R1:.*]]: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>,
    // CHECK-SAME: [%[[STATE_ARG0]], %[[STATE_ARG1]]] as %[[R_STATE:.*]]: tensor<!tf.resource<tensor<2x!tf.string>>>
    %rep:2 = tf_device.replicate([%arg1, %arg2] as %arg30: tensor<*x!tf.resource<tensor<f32>>>,
                        [%arg3, %arg4] as %arg31: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>)
            {_mirrored_variable_indices = [0, 1], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
      // CHECK: %[[ID:.*]] = "tf.Identity"(%[[R0]])
      %id = "tf.Identity"(%arg30) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource<tensor<f32>>>
      // CHECK: "tf.TPUReshardVariables"(%[[ID]], %[[R1]], %[[COMPILE]]#1, %[[R_STATE]])
      // CHECK: "tf.TPUExecuteAndUpdateVariables"(%[[ID]], %[[R1]], %[[COMPILE]]#1)
      "tf.TPUExecuteAndUpdateVariables"(%id, %arg31, %2#1)
            {device_var_reads_indices = [0, 1], device_var_updates_indices = [0, 1]}
              : (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<!tf.string>) -> ()
      %ret = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      tf_device.return %ret : tensor<i32>
    }
    return %1, %arg1, %arg2, %arg3, %arg4 : tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>,
              tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>,
              tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>
  }
  // CHECK-LABEL: func @while_cond_7550
  func @while_cond_7550(%arg0: tensor<i32>,
                        %arg1: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
                        %arg2: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
                        %arg3: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
                        %arg4: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"})
       -> tensor<i1> {
    %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.GreaterEqual"(%arg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}

// -----

// Tests that the pass does not format variabls with other uses.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: func @main
  // CHECK-NOT: TPUReshardVariables
  func @main(%arg0: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
             %arg1: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
             %arg2: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
             %arg3: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"}) {
    %0 = "tf.Const"() {value = dense<100> : tensor<i32>} : () -> tensor<i32>
    %1:5 = "tf.While"(%0, %arg0, %arg1, %arg2, %arg3)
               {T = ["tfdtype$DT_INT32", "tfdtype$DT_RESOURCE",
                 "tfdtype$DT_RESOURCE", "tfdtype$DT_RESOURCE",
                 "tfdtype$DT_RESOURCE"], body = @while_body_7560,
                cond = @while_cond_7550, device = "", is_stateless = false,
                output_shapes = ["tfshape$", "tfshape$", "tfshape$", "tfshape$", "tfshape$"]}
         : (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>,
            tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>)
         -> (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>,
             tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>)
    return
  }
  // CHECK: func @while_body_7560
  // CHECK-NOT: TPUReshardVariables
  func @while_body_7560(%arg0: tensor<i32>,
                        %arg1: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
                        %arg2: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
                        %arg3: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
                        %arg4: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"})
        -> (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>,
            tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>) {
    %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.AddV2"(%arg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64, device = "/device:CPU:0",
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    "tf.TPUCompileSucceededAssert"(%2#0) : (tensor<!tf.string>) -> ()
    %new_var = "tf._UnknownOp0_"(%arg3) : (tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>) -> tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>
    tf_device.replicate([%arg1, %arg2] as %arg30: tensor<*x!tf.resource<tensor<f32>>>,
                        [%new_var, %arg4] as %arg31: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>)
            {_mirrored_variable_indices = [0, 1], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
      // %arg30 is used in the cond function, and %arg31 is not pass-through of
      // while inputs, so neither should be formatted.
      "tf.TPUExecuteAndUpdateVariables"(%arg30, %arg31, %2#1)
            {device_var_reads_indices = [0, 1], device_var_updates_indices = [0, 1]}
              : (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>, tensor<!tf.string>) -> ()
      tf_device.return
    }
    return %1, %arg1, %arg2, %arg3, %arg4 : tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>,
              tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>,
              tensor<*x!tf.resource<tensor<3x3x1x32xf32>>>
  }
  // CHECK-LABEL: func @while_cond_7550
  func @while_cond_7550(%arg0: tensor<i32>,
                        %arg1: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
                        %arg2: tensor<*x!tf.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
                        %arg3: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
                        %arg4: tensor<*x!tf.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"})
       -> tensor<i1> {
    %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.GreaterEqual"(%arg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tf._UnknownOp1_"(%arg1) : (tensor<*x!tf.resource<tensor<f32>>>) -> ()
    return %1 : tensor<i1>
  }
}
