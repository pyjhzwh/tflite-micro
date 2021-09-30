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


TF_LITE_MICRO_TEST(TestTopologicalMedium) {
  tflite::MicroErrorReporter micro_error_reporter;
  // 0              1                   2                  3               4          
  // buffer0 -> conv2d -> buffer1 -> conv2d -> buffer2 -> add -> buffer 4
  // buffer3 ----------------------------------------------|
  tflite::TopologicalMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize, 3);

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

  tflite::OpParams conv2dParams2;
  conv2dParams2.convOpParams.input_height = 3;
  conv2dParams2.convOpParams.input_width = 3;
  conv2dParams2.convOpParams.input_channel = 5;
  conv2dParams2.convOpParams.filter_height = 3;
  conv2dParams2.convOpParams.filter_width = 3;
  conv2dParams2.convOpParams.output_height = 3;
  conv2dParams2.convOpParams.output_width = 3;
  conv2dParams2.convOpParams.output_channel = 3;
  conv2dParams2.convOpParams.padding_height = 1;
  conv2dParams2.convOpParams.padding_width = 1;
  conv2dParams2.convOpParams.padding_height_offset = 0;
  conv2dParams2.convOpParams.padding_width_offset = 0;
  conv2dParams2.convOpParams.stride_height = 1;
  conv2dParams2.convOpParams.stride_width = 1;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 1, 
                                                tflite::BuiltinOperator_CONV_2D, &conv2dParams2));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 2, 
                                                tflite::BuiltinOperator_ADD, nullptr));                                              

  bool input_of_operators_buffer0[3] = {1,0,0};                                              
  bool output_of_operators_buffer0[3] = {0,0,0};             
  bool input_of_operators_buffer1[3] = {0,1,0};                                              
  bool output_of_operators_buffer1[3] = {1,0,0};        
  bool input_of_operators_buffer2[3] = {0,0,1};                                              
  bool output_of_operators_buffer2[3] = {0,1,0};                   
  bool input_of_operators_buffer3[3] = {0,0,1};                                              
  bool output_of_operators_buffer3[3] = {0,0,0};    
  bool input_of_operators_buffer4[3] = {0,0,0};                                              
  bool output_of_operators_buffer4[3] = {0,0,1};                 
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3*3*3, 0, 1, 
                                            input_of_operators_buffer0,
                                            output_of_operators_buffer0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3*3*5, 1, 2, 
                                            input_of_operators_buffer1,
                                            output_of_operators_buffer1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3*3*3, 2, 3, 
                                            input_of_operators_buffer2,
                                            output_of_operators_buffer2)); 
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3*3*3, 0, 3, 
                                            input_of_operators_buffer3,
                                            output_of_operators_buffer3));         
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 3*3*3, 3, 4, 
                                            input_of_operators_buffer4,
                                            output_of_operators_buffer4));                                                                                                                  


  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 0, &offset));
  TF_LITE_MICRO_EXPECT_EQ(3*3*3, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 1, &offset));
  TF_LITE_MICRO_EXPECT_EQ(3*3*3+15, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 2, &offset));
  TF_LITE_MICRO_EXPECT_EQ(3*3*3, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 3, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 4, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  planner.PrintMemoryPlan();

  TF_LITE_MICRO_EXPECT_EQ(true,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(60+3*3*3),
                          planner.GetMaximumMemorySize());
}


TF_LITE_MICRO_TEST(TestAllCNNNetModel) {
  tflite::MicroErrorReporter micro_error_reporter;

  // conv0 - conv8
  tflite::TopologicalMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize, 9);

  tflite::ConvOpParams conv0Params = {/*.padding_type*/ tflite::PaddingType::kSame, 
      /*padding_height*/ 1, /*padding_width*/ 1,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 1, /*stride_height*/ 1,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 32, /*input_width*/ 32,
      /*input_channel*/ 3, /*filter_height*/ 3, /*filter_width*/ 3, /*output_height*/ 32,
      /*output_width*/ 32, /*output_channel*/ 96};

  tflite::ConvOpParams conv1Params = {/*.padding_type*/ tflite::PaddingType::kSame, 
      /*padding_height*/ 1, /*padding_width*/ 1,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 1, /*stride_height*/ 1,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 32, /*input_width*/ 32,
      /*input_channel*/ 96, /*filter_height*/ 3, /*filter_width*/ 3, /*output_height*/ 32,
      /*output_width*/ 32, /*output_channel*/ 96};

  tflite::ConvOpParams conv2Params = {/*.padding_type*/ tflite::PaddingType::kValid, 
      /*padding_height*/ 1, /*padding_width*/ 1,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 2, /*stride_height*/ 2,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 32, /*input_width*/ 32,
      /*input_channel*/ 96, /*filter_height*/ 3, /*filter_width*/ 3, /*output_height*/ 16,
      /*output_width*/ 16, /*output_channel*/ 96};
  
  tflite::ConvOpParams conv3Params = {/*.padding_type*/ tflite::PaddingType::kSame, 
      /*padding_height*/ 1, /*padding_width*/ 1,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 1, /*stride_height*/ 1,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 16, /*input_width*/ 16,
      /*input_channel*/ 96, /*filter_height*/ 3, /*filter_width*/ 3, /*output_height*/ 16,
      /*output_width*/ 16, /*output_channel*/ 192};

  tflite::ConvOpParams conv4Params = {/*.padding_type*/ tflite::PaddingType::kSame, 
      /*padding_height*/ 1, /*padding_width*/ 1,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 1, /*stride_height*/ 1,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 16, /*input_width*/ 16,
      /*input_channel*/ 192, /*filter_height*/ 3, /*filter_width*/ 3, /*output_height*/ 16,
      /*output_width*/ 16, /*output_channel*/ 192};
  
  tflite::ConvOpParams conv5Params = {/*.padding_type*/ tflite::PaddingType::kValid, 
      /*padding_height*/ 1, /*padding_width*/ 1,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 2, /*stride_height*/ 2,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 16, /*input_width*/ 16,
      /*input_channel*/ 192, /*filter_height*/ 3, /*filter_width*/ 3, /*output_height*/ 8,
      /*output_width*/ 8, /*output_channel*/ 192};

  tflite::ConvOpParams conv6Params = {/*.padding_type*/ tflite::PaddingType::kSame, 
      /*padding_height*/ 1, /*padding_width*/ 1,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 1, /*stride_height*/ 1,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 8, /*input_width*/ 8,
      /*input_channel*/ 192, /*filter_height*/ 3, /*filter_width*/ 3, /*output_height*/ 8,
      /*output_width*/ 8, /*output_channel*/ 192};

  tflite::ConvOpParams conv7Params = {/*.padding_type*/ tflite::PaddingType::kNone, 
      /*padding_height*/ 0, /*padding_width*/ 0,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 1, /*stride_height*/ 1,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 8, /*input_width*/ 8,
      /*input_channel*/ 192, /*filter_height*/ 1, /*filter_width*/ 1, /*output_height*/ 8,
      /*output_width*/ 8, /*output_channel*/ 192};

  tflite::ConvOpParams conv8Params = {/*.padding_type*/ tflite::PaddingType::kNone, 
      /*padding_height*/ 0, /*padding_width*/ 0,
      /*padding_height_offset*/ 0, /*padding_width_offset*/ 0, /*stride_width*/ 1, /*stride_height*/ 1,
      /*dilation_width_factor*/ 1, /*dilation_height_factor*/ 1, /*input_height*/ 8, /*input_width*/ 8,
      /*input_channel*/ 192, /*filter_height*/ 1, /*filter_width*/ 1, /*output_height*/ 8,
      /*output_width*/ 8, /*output_channel*/ 10};

  tflite::OpParams conv0;
  tflite::OpParams conv1;
  tflite::OpParams conv2;
  tflite::OpParams conv3;
  tflite::OpParams conv4;
  tflite::OpParams conv5;
  tflite::OpParams conv6;
  tflite::OpParams conv7;
  tflite::OpParams conv8;

  conv0.convOpParams = conv0Params;
  conv1.convOpParams = conv1Params;
  conv2.convOpParams = conv2Params;
  conv3.convOpParams = conv3Params;
  conv4.convOpParams = conv4Params;
  conv5.convOpParams = conv5Params;
  conv6.convOpParams = conv6Params;
  conv7.convOpParams = conv7Params;
  conv8.convOpParams = conv8Params;
  

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 0, 
                                                tflite::BuiltinOperator_CONV_2D, &conv0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 1, 
                                                tflite::BuiltinOperator_CONV_2D, &conv1));                                                
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 2, 
                                                tflite::BuiltinOperator_CONV_2D, &conv2));        
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 3, 
                                                tflite::BuiltinOperator_CONV_2D, &conv3));     
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 4, 
                                                tflite::BuiltinOperator_CONV_2D, &conv4));     
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 5, 
                                                tflite::BuiltinOperator_CONV_2D, &conv5));                                                                                         
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 6, 
                                                tflite::BuiltinOperator_CONV_2D, &conv6));     
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 7, 
                                                tflite::BuiltinOperator_CONV_2D, &conv7));   
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 8, 
                                                tflite::BuiltinOperator_CONV_2D, &conv8));      

  // 9 layers, 10 buffers
  bool input_of_operators[10][9] = {false};                                           
  bool output_of_operators[10][9] = {false};     
  input_of_operators[0][0] = true;
  output_of_operators[9][8] = true;
  for(int i=1; i <9; i++) {
    input_of_operators[i][i] = true;
    output_of_operators[i][i-1] = true;
  }             
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv0Params.input_height * conv0Params.input_width *conv0Params.input_channel,
                              0, 1, input_of_operators[0], output_of_operators[0]));                    
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv1Params.input_height * conv1Params.input_width *conv1Params.input_channel,
                              1, 2, input_of_operators[1], output_of_operators[1]));                         
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv2Params.input_height * conv2Params.input_width *conv2Params.input_channel,
                              2, 3, input_of_operators[2], output_of_operators[2]));                         
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv3Params.input_height * conv3Params.input_width *conv3Params.input_channel,
                              3, 4, input_of_operators[3], output_of_operators[3]));                    
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv4Params.input_height * conv4Params.input_width *conv4Params.input_channel,
                              4, 5, input_of_operators[4], output_of_operators[4]));                         
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv5Params.input_height * conv5Params.input_width *conv5Params.input_channel,
                              5, 6, input_of_operators[5], output_of_operators[5]));                         
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv6Params.input_height * conv6Params.input_width *conv6Params.input_channel,
                              6, 7, input_of_operators[6], output_of_operators[6]));                    
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv7Params.input_height * conv7Params.input_width *conv7Params.input_channel,
                              7, 8, input_of_operators[7], output_of_operators[7]));                         
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv8Params.input_height * conv8Params.input_width *conv8Params.input_channel,
                              8, 9, input_of_operators[8], output_of_operators[8]));                         

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 
                              conv8Params.output_height * conv8Params.output_width *conv8Params.output_channel,
                              9, 10, input_of_operators[9], output_of_operators[9]));

  planner.PrintMemoryPlan();

  TF_LITE_MICRO_EXPECT_EQ(true,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 0, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 1, &offset));
  TF_LITE_MICRO_EXPECT_EQ(102, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 2, &offset));
  TF_LITE_MICRO_EXPECT_EQ(3366, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 3, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 4, &offset));
  TF_LITE_MICRO_EXPECT_EQ(1728, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 5, &offset));
  TF_LITE_MICRO_EXPECT_EQ(5184, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 6, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 7, &offset));
  TF_LITE_MICRO_EXPECT_EQ(1920, offset);

   TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 8, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(&micro_error_reporter, 9, &offset));
  TF_LITE_MICRO_EXPECT_EQ(11658, offset);

 
 
  // The results in self_convolution_mem.py is 101670 B
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(101670),
                          planner.GetMaximumMemorySize());                            
}
TF_LITE_MICRO_TEST(TestNoOverlapCase) {
  tflite::MicroErrorReporter micro_error_reporter;

  tflite::TopologicalMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize, 2);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 0, 
                                                tflite::BuiltinOperator_MUL, nullptr));                                              
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 1, 
                                                tflite::BuiltinOperator_MUL, nullptr));                                              

  bool input_of_operators_buffer0[3] = {1,0};                                              
  bool output_of_operators_buffer0[3] = {0,0};             
  bool input_of_operators_buffer1[3] = {0,1};                                              
  bool output_of_operators_buffer1[3] = {1,0};        
  bool input_of_operators_buffer2[3] = {0,0};                                              
  bool output_of_operators_buffer2[3] = {0,1};    

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 100, 0, 1,
                                            input_of_operators_buffer0,
                                            output_of_operators_buffer0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 50, 2, 3, 
                                            input_of_operators_buffer1,
                                            output_of_operators_buffer1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 20, 1, 2, 
                                            input_of_operators_buffer2,
                                            output_of_operators_buffer2));

  planner.PrintMemoryPlan();

  TF_LITE_MICRO_EXPECT_EQ(false,
                          planner.DoAnyBuffersOverlap(&micro_error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(120),
                          planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(TestSmallScratch) {
  tflite::MicroErrorReporter micro_error_reporter;

  constexpr int scratch_buffer_size = 200;
  unsigned char scratch_buffer[scratch_buffer_size];
  tflite::TopologicalMemoryPlanner planner(scratch_buffer, scratch_buffer_size, 1);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddOperatorInfo(&micro_error_reporter, 0, 
                                                tflite::BuiltinOperator_MUL, nullptr));                                                                                   

  bool input_of_operators_buffer0[1] = {1};                                              
  bool output_of_operators_buffer0[1] = {0};             
  bool input_of_operators_buffer1[1] = {0};                                              
  bool output_of_operators_buffer1[1] = {1};        

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 100, 0, 1,
                                            input_of_operators_buffer0,
                                            output_of_operators_buffer0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(&micro_error_reporter, 50, 2, 3, 
                                            input_of_operators_buffer1,
                                            output_of_operators_buffer1));
}

TF_LITE_MICRO_TESTS_END
