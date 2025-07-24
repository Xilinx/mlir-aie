
func.func @test_matmul(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %0 =  "onnx.MatMul"(%arg0, %arg1) : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32> 
  return %0 : tensor<1024x1024xf32>
}
