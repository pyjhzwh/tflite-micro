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

#include "tensorflow/lite/micro/memory_planner/topological_memory_planner.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
// We don't declare this in the header since it's not a public interface, but we
// need to call it to test it, so declare it here instead.
void SortInPlace2Level(int* val1s, int* val2s, int* ids, int size);
}  // namespace tflite

namespace {
constexpr int kScratchBufferSize = 4096;
unsigned char g_scratch_buffer[kScratchBufferSize];
}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

// First level: ascending order of values1
// Second level: for the same values1, descending order of values2
TF_LITE_MICRO_TEST(TestSortInPlace2Level) {
  tflite::MicroErrorReporter micro_error_reporter;

  constexpr int a_size = 10;
  int a_val1s[a_size] = {1, 2, 2, 3, 4, 5, 6, 7, 8, 9};
  int a_val2s[a_size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int a_ids[a_size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // a_val1s and a_val2s is already sorted, no need to swap 
  const int a_expected_val1s[a_size] = {1, 2, 2, 3, 4, 5, 6, 7, 8, 9};
  const int a_expected_val2s[a_size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  const int a_expected_ids[a_size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  tflite::SortInPlace2Level(a_val1s, a_val2s, a_ids, a_size);
  for (int i = 0; i < a_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(a_expected_val1s[i], a_val1s[i]);
    TF_LITE_MICRO_EXPECT_EQ(a_expected_val2s[i], a_val2s[i]);
    TF_LITE_MICRO_EXPECT_EQ(a_expected_ids[i], a_ids[i]);
  }
  
  constexpr int b_size = 10;
  int b_val1s[b_size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int b_val2s[b_size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int b_ids[b_size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // should reverse b to get expected vals and ids
  const int b_expected_val1s[b_size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int b_expected_val2s[b_size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int b_expected_ids[b_size] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  tflite::SortInPlace2Level(b_val1s, b_val2s, b_ids, b_size);
  for (int i = 0; i < b_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(b_expected_val1s[i], b_val1s[i]);
    TF_LITE_MICRO_EXPECT_EQ(b_expected_val2s[i], b_val2s[i]);
    TF_LITE_MICRO_EXPECT_EQ(b_expected_ids[i], b_ids[i]);
  }

  constexpr int c_size = 100;
  int c_val1s[c_size] = {
      10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
      10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
      10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
      10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
      10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int c_val2s[c_size] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
      69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
      86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100};
  int c_ids[c_size] = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
      51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
      68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
      85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
  const int c_expected_val1s[c_size] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
      9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
  const int c_expected_val2s[c_size] = {
      100, 90, 80, 70, 60, 50, 40, 30, 20, 10,
      99, 89, 79, 69, 59, 49, 39, 29, 19, 9,
      98, 88, 78, 68, 58, 48, 38, 28, 18, 8,
      97, 87, 77, 67, 57, 47, 37, 27, 17, 7,
      96, 86, 76, 66, 56, 46, 36, 26, 16, 6,
      95, 85, 75, 65, 55, 45, 35, 25, 15, 5,
      94, 84, 74, 64, 54, 44, 34, 24, 14, 4,
      93, 83, 73, 63, 53, 43, 33, 23, 13, 3,
      92, 82, 72, 62, 52, 42, 32, 22, 12, 2,  
      91, 81, 71, 61, 51, 41, 31, 21, 11, 1      
      };
  const int c_expected_ids[c_size] = {
      99, 89, 79, 69, 59, 49, 39, 29, 19, 9,
      98, 88, 78, 68, 58, 48, 38, 28, 18, 8,
      97, 87, 77, 67, 57, 47, 37, 27, 17, 7,
      96, 86, 76, 66, 56, 46, 36, 26, 16, 6,
      95, 85, 75, 65, 55, 45, 35, 25, 15, 5,
      94, 84, 74, 64, 54, 44, 34, 24, 14, 4,
      93, 83, 73, 63, 53, 43, 33, 23, 13, 3,
      92, 82, 72, 62, 52, 42, 32, 22, 12, 2,
      91, 81, 71, 61, 51, 41, 31, 21, 11, 1,
      90, 80, 70, 60, 50, 40, 30, 20, 10, 0};
  tflite::SortInPlace2Level(c_val1s, c_val2s, c_ids, c_size);
  for (int i = 0; i < c_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(c_expected_val1s[i], c_val1s[i]);
    TF_LITE_MICRO_EXPECT_EQ(c_expected_val2s[i], c_val2s[i]);
    TF_LITE_MICRO_EXPECT_EQ(c_expected_ids[i], c_ids[i]);
  }
  
}

TF_LITE_MICRO_TEST(TestTopologicalBasics) {
  tflite::MicroErrorReporter micro_error_reporter;

  tflite::TopologicalMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize, 1);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 0, 
                                                tflite::BuiltinOperator_MUL, nullptr));
  bool input_of_operators_buffer0[1] = {1};                                              
  bool output_of_operators_buffer0[1] = {0};             
  bool input_of_operators_buffer1[1] = {0};                                              
  bool output_of_operators_buffer1[1] = {1};                                  
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 10, 0, 1, 
                                            input_of_operators_buffer0,
                                            output_of_operators_buffer0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 20, 2, 3, 
                                            input_of_operators_buffer1,
                                            output_of_operators_buffer1));

  TF_LITE_MICRO_EXPECT_EQ(false,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(20),
                          planner.GetMaximumMemorySize());

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 0, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 1, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);
}

TF_LITE_MICRO_TEST(TestTopologicalBasicsConv) {
  tflite::MicroErrorReporter micro_error_reporter;

  tflite::TopologicalMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize, 1);
  tflite::OpParams conv2dParams;
  conv2dParams.convOpParams.input_height = 3;
  conv2dParams.convOpParams.input_width = 3;
  conv2dParams.convOpParams.input_channel = 3;
  conv2dParams.convOpParams.filter_height = 3;
  conv2dParams.convOpParams.filter_width = 3;
  conv2dParams.convOpParams.output_height = 3;
  conv2dParams.convOpParams.output_width = 3;
  conv2dParams.convOpParams.output_channel = 5;
  conv2dParams.convOpParams.padding_height = 1;
  conv2dParams.convOpParams.padding_width = 1;
  conv2dParams.convOpParams.padding_height_offset = 0;
  conv2dParams.convOpParams.padding_width_offset = 0;
  conv2dParams.convOpParams.stride_height = 1;
  conv2dParams.convOpParams.stride_width = 1;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 0, 
                                                tflite::BuiltinOperator_CONV_2D, &conv2dParams));
  bool input_of_operators_buffer0[1] = {1};                                              
  bool output_of_operators_buffer0[1] = {0};             
  bool input_of_operators_buffer1[1] = {0};                                              
  bool output_of_operators_buffer1[1] = {1};                                  
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3*3*3, 0, 1, 
                                            input_of_operators_buffer0,
                                            output_of_operators_buffer0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3*3*5, 1, 2, 
                                            input_of_operators_buffer1,
                                            output_of_operators_buffer1));

  TF_LITE_MICRO_EXPECT_EQ(true,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(60),
                          planner.GetMaximumMemorySize());

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 0, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 1, &offset));
  TF_LITE_MICRO_EXPECT_EQ(15, offset);
}

/*
TF_LITE_MICRO_TEST(TestGreedyMedium) {
  tflite::MicroErrorReporter micro_error_reporter;

  tflite::GreedyMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 10, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 20, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 30, 2, 3));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 40, 3, 4));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 50, 0, 1));

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 0, &offset));
  TF_LITE_MICRO_EXPECT_EQ(50, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 1, &offset));
  TF_LITE_MICRO_EXPECT_EQ(70, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 2, &offset));
  TF_LITE_MICRO_EXPECT_EQ(40, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 3, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 4, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  planner.PrintMemoryPlan();

  TF_LITE_MICRO_EXPECT_EQ(false,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(90),
                          planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(TestPersonDetectionModel) {
  tflite::MicroErrorReporter micro_error_reporter;

  tflite::GreedyMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize);
  // These buffer sizes and time ranges are taken from the 250KB MobileNet model
  // used in the person detection example.
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 9216, 0, 29));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3, 28, 29));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 256, 27, 28));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 2304, 26, 27));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 2304, 25, 26));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 2304, 24, 25));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 1152, 23, 24));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 22, 23));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 21, 22));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 20, 21));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 19, 20));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 18, 19));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 17, 18));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 16, 17));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 15, 16));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 14, 15));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 13, 14));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 4608, 12, 13));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 2304, 11, 12));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 9216, 10, 11));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 9216, 9, 10));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 9216, 8, 9));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 4608, 7, 8));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 18432, 6, 7));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 18432, 5, 6));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 18432, 4, 5));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 9216, 3, 4));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 36864, 2, 3));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 18432, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.AddBuffer(&micro_error_reporter, 18432, 0, 1));

  planner.PrintMemoryPlan();

  TF_LITE_MICRO_EXPECT_EQ(false,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  // The sum of all the buffers is 241,027 bytes, so we at least expect the plan
  // to come up with something smaller than this.
  TF_LITE_MICRO_EXPECT_GT(static_cast<size_t>(241027),
                          planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(TestOverlapCase) {
  tflite::MicroErrorReporter micro_error_reporter;

  tflite::GreedyMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 100, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 50, 2, 3));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 20, 1, 2));

  planner.PrintMemoryPlan();

  TF_LITE_MICRO_EXPECT_EQ(false,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(120),
                          planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(TestSmallScratch) {
  tflite::MicroErrorReporter micro_error_reporter;

  constexpr int scratch_buffer_size = 40;
  unsigned char scratch_buffer[scratch_buffer_size];
  tflite::GreedyMemoryPlanner planner(scratch_buffer, scratch_buffer_size);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 100, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          planner.AddBuffer(&micro_error_reporter, 50, 2, 3));
}
*/
TF_LITE_MICRO_TESTS_END
