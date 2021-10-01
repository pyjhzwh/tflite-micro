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

#include "tensorflow/lite/micro/micro_allocator.h"

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

#define TOPOLOGY_MEM_PLANNER
#define TF_LITE_SHOW_MEMORY_USE

namespace tflite {
namespace testing {
namespace {

constexpr int kExpectedAlignment = 4;
constexpr int t0 = 0;
constexpr int t1 = 1;
constexpr int t2 = 2;
constexpr int t3 = 3;
constexpr int t4 = 4;
constexpr int t5 = 5;

void VerifyMockConvTfLiteTensor(TfLiteTensor* tensor, bool is_variable = false) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(4, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(is_variable, tensor->is_variable);
  //TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), tensor->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          (reinterpret_cast<std::uintptr_t>(tensor->data.raw) %
                           kExpectedAlignment));
}

void VerifyMockConvWeightTfLiteTensor(TfLiteTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(4, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(5, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(5*3*3*3), tensor->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
}

void VerifyMockConvTfLiteEvalTensor(TfLiteEvalTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(4, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(tensor, &buffer_size));
  //TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), buffer_size);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          (reinterpret_cast<std::uintptr_t>(tensor->data.raw) %
                           kExpectedAlignment));
}

void VerifyMockConvWeightTfLiteEvalTensor(TfLiteEvalTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(4, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(5, tensor->dims->data[0]);
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(tensor, &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(5*3*3*3), buffer_size);
  TF_LITE_MICRO_EXPECT_NE(nullptr, tensor->data.raw);
}

void VerifyMockConvTensor(const Model* model, MicroAllocator* allocator,
                      SubgraphAllocations* subgraph_allocations, int tensor_idx,
                      bool is_variable = false) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    VerifyMockConvTfLiteTensor(
        allocator->AllocatePersistentTfLiteTensor(model, subgraph_allocations,
                                                  tensor_idx, subgraph_idx),
        is_variable);
    VerifyMockConvTfLiteEvalTensor(
        &subgraph_allocations[subgraph_idx].tensors[tensor_idx]);
  }
}

void VerifyMockConvWeightTensor(const Model* model, MicroAllocator* allocator,
                            SubgraphAllocations* subgraph_allocations,
                            int tensor_idx) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    VerifyMockConvWeightTfLiteTensor(allocator->AllocatePersistentTfLiteTensor(
        model, subgraph_allocations, tensor_idx, subgraph_idx));
    VerifyMockConvWeightTfLiteEvalTensor(
        &subgraph_allocations[subgraph_idx].tensors[tensor_idx]);
  }
}


void VerifyRegistrationAndNodeAllocation(
    SubgraphAllocations* subgraph_allocations, size_t count,
    int num_subgraphs) {
  for (int subgraph_idx = 0; subgraph_idx < num_subgraphs; subgraph_idx++) {
    for (size_t i = 0; i < count; i++) {
      TF_LITE_MICRO_EXPECT_NE(nullptr, &subgraph_allocations[subgraph_idx]
                                            .node_and_registrations[i]
                                            .registration);
    }
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN


TF_LITE_MICRO_TEST(TestMockConvModelAllocation) {
  const tflite::Model* model = tflite::testing::GetSimpleMockConvModel();
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::AllOpsResolver op_resolver = tflite::testing::GetOpResolver();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      arena, arena_size, tflite::GetMicroErrorReporter());
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(3), model_tensor_size);

  tflite::testing::VerifyMockConvTensor(model, allocator, subgraph_allocations, 0);
  tflite::testing::VerifyMockConvWeightTensor(model, allocator,
                                          subgraph_allocations, 1);
  tflite::testing::VerifyMockConvTensor(model, allocator, subgraph_allocations, 2);

  TfLiteEvalTensor* eval_tensors = subgraph_allocations[0].tensors;
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[2].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[2].data.raw);

  TF_LITE_MICRO_EXPECT_LT(static_cast<int>(allocator->used_bytes()), 776 + 100); // TODO: size should be

  // SimpleMockModel has 2 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(subgraph_allocations,
                                                       /*count=*/1,
                                                       /*num_subgraphs=*/1);
}


TF_LITE_MICRO_TESTS_END
