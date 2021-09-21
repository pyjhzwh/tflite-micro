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
#include "tensorflow/lite/micro/micro_string.h"

namespace tflite {

namespace {

// Returns a character representing a numbered buffer
// for TopologicalMemoryPlanner::PrintMemoryPlan()
char GetOrdinalCharacter(int i) {
  if (i < 10) {
    return '0' + i;
  } else if (i < 36) {
    return 'a' + (i - 10);
  } else if (i < 62) {
    return 'A' + (i - 36);
  }
  return '*';
}

}  // namespace

// Simple stable in-place sort function. Not time-efficient for large arrays.
// Would normally be in an anonymous namespace to keep it private, but we want
// to be able to test it externally.
void SortInPlace(int* values, int* ids, int size) {
  bool any_swapped;
  do {
    any_swapped = false;
    for (int i = 1; i < size; ++i) {
      if (values[i - 1] > values[i]) {
        const int value_temp = values[i - 1];
        values[i - 1] = values[i];
        values[i] = value_temp;
        const int id_temp = ids[i - 1];
        ids[i - 1] = ids[i];
        ids[i] = id_temp;
        any_swapped = true;
      }
    }
  } while (any_swapped);
}

int calculatePaddingLen(OperatorRequirements* op_requirement, void* node_params, void* op_data) {

  BuiltinOperator op_type = op_requirement->op_type;
  // if node is conv2d
  if (op_type == BuiltinOperator_CONV_2D) {
    
  }
  // if node is in-place operation
  else if (op_type == BuiltinOperator_ADD)
  {
    return 0;
  }
  

}

int conv2DMemAllocation(Conv2DOpParams* op_params) {
  // create depedency
  for (int out_hi = op_params->output_height; out_hi >= 0; --out_hi) {
    for (int out_wi = op_params->output_width; out_wi >= 0; --out_wi) {
      // find the corresponding input window
      // add dependencies
      int in_hi = out_hi * op_params->stride_height - op_params->padding
      int in_wi = out_wi * op_params->stride_width - op_params->padding
      // only keep the last dependent out tensor 
      for (int i = 0; i < op_params->filter_height; ++i) {
        for (int j = 0; j < op_params->filter_width; ++j) {
          int x = in_hi + i
          int y = in_wi + j
          // if not assigned child
          if (x >= 0 && y >= 0 && x < h && y < w && last_child_of_tensor[x][y]==-1) {
            // TODO: use struct not tuple in python
            last_child_of_tensor[x][y] = 
                (out_hi * op_params->output_width + out_wi) * op_params->output_channel;
          }
          // otherwise, continue because we only care last child
        }
      }
    }
  }
  // calculate actual memory size
  curend = 0
  for (int in_hi = 0; in_hi < op_params->input_height; ++in_hi) {
    for (int in_wi = 0; in_wi < op_params->input_width; ++in_wi) {
        // TODO: use struct not tuple in python
        outmem_pos_lastchild = last_child_of_tensor[in_hi][in_wi]
        curend = max(curend, outmem_pos_lastchild)
        curend += op_params->input_channel;
    }
  }

  return curend - self.input_height * self.input_width * self.input_channel;
}

TopologicalMemoryPlanner::TopologicalMemoryPlanner(unsigned char* scratch_buffer,
                                         int scratch_buffer_size, int operator_size)
    : buffer_count_(0), need_to_calculate_offsets_(true) {
  // Allocate the arrays we need within the scratch buffer arena.
  max_buffer_count_ = (scratch_buffer_size - sizeof(OperatorInfo) * operator_size) / 
                      (per_buffer_size() + sizeof(int) * 2 * operator_size_);
  operator_size_ = operator_size;

  unsigned char* next_free = scratch_buffer;
  requirements_ = reinterpret_cast<BufferRequirements*>(next_free);
  next_free += sizeof(BufferRequirements) * max_buffer_count_;

  // Allocate bool arrays for each requirements_
  for (int i=0; i < operator_size; ++i) {
    requirements_[i].input_of_operators = reinterpret_cast<bool*>(next_free);
    next_free += sizeof(bool) * operator_size;
    requirements_[i].output_of_operators = reinterpret_cast<bool*>(next_free);
    next_free += sizeof(bool) * operator_size;
  }

  //buffer_created_sorted_ = reinterpret_cast<int*>(next_free);
  //next_free += sizeof(int) * max_buffer_count_;

  //buffer_ids_sorted_ = reinterpret_cast<int*>(next_free);
  //next_free += sizeof(int) * max_buffer_count_;

  /*
  buffers_sorted_by_offset_ = reinterpret_cast<ListEntry*>(next_free);
  next_free += sizeof(ListEntry) * max_buffer_count_;
  */
  // Allocate space for struct of Operator
  operator_info_ = reinterpret_cast<OperatorInfo*>(next_free);
  next_free += sizeof(OperatorInfo) * operator_size_;
  // Allocate space to store int* of inputs and outputs of operatos
  next_free += sizeof(int) * max_buffer_count_ * 2 * operator_size_;

  buffer_offsets_ = reinterpret_cast<int*>(next_free);
}

TopologicalMemoryPlanner::~TopologicalMemoryPlanner() {
  // We don't own the scratch buffer, so don't deallocate anything.
}

TfLiteStatus TopologicalMemoryPlanner::AddOperatorInfo(
  tflite::ErrorReporter* error_reporter, int operator_id, BuiltinOperator op_type, 
  OpParams* op_params) {
    if (operator_id >= operator_size_) {
      TF_LITE_REPORT_ERROR(error_reporter, "Operator index larger than size (%d)",
                         operator_size_);
      return kTfLiteError;
    }
    OperatorInfo* current_op = &operator_info_[operator_id];
    current_op->op_type = op_type;
    switch(op_type) {
      case BuiltinOperator_CONV_2D:
        ConvOpParams* current_op_params = &(current_op->params->convOpParams);
        ConvOpParams* input_op_params = &(op_params->convOpParams);
        current_op_params->input_height = input_op_params->input_height;
        current_op_params->input_width = input_op_params->input_width;
        current_op_params->input_channel = input_op_params->input_channel;
        current_op_params->filter_height = input_op_params->filter_height;
        current_op_params->filter_width = input_op_params->filter_width;
        current_op_params->output_height = input_op_params->output_height;
        current_op_params->output_width = input_op_params->output_width;
        current_op_params->output_channel = input_op_params->output_channel;
        current_op_params->padding = input_op_params->padding;
        current_op_params->stride_width = input_op_params->stride_width;
        current_op_params->stride_height = input_op_params->stride_height;
        current_op_params->dilation_width_factor = input_op_params->dilation_width_factor;
        current_op_params->dilation_height_factor = input_op_params->dilation_height_factor;
        break;

      default:
        break;
    }
    
    return kTfLiteOk;
}

TfLiteStatus TopologicalMemoryPlanner::AddBuffer(
    tflite::ErrorReporter* error_reporter, int size, int first_time_used,
    int last_time_used, bool* input_of_operators, bool* output_of_operators,
    int operator_size) {
  if (buffer_count_ >= max_buffer_count_) {
    TF_LITE_REPORT_ERROR(error_reporter, "Too many buffers (max is %d)",
                         max_buffer_count_);
    return kTfLiteError;
  }
  BufferRequirements* current = &requirements_[buffer_count_];
  current->size = size;
  current->first_time_used = first_time_used;
  current->last_time_used = last_time_used;
  current->offline_offset = kOnlinePlannedBuffer;
  // deep copy of input_of_operators and output_of_operators arrays;
  bool* current_input_of_operators = current->input_of_operators;
  for (int i = 0; i < operator_size; ++i) {
    current_input_of_operators[i] = input_of_operators[i];
  }
  bool* current_output_of_operators = current->output_of_operators;
  for (int i = 0; i < operator_size; ++i) {
    current_output_of_operators[i] = output_of_operators[i];
  }
  ++buffer_count_;
  need_to_calculate_offsets_ = true;
  return kTfLiteOk;
}

TfLiteStatus TopologicalMemoryPlanner::AddBuffer(
    tflite::ErrorReporter* error_reporter, int size, int first_time_used,
    int last_time_used, bool* input_of_operators, bool* output_of_operators,
    int operator_size, int offline_offset) {
  BufferRequirements* current = &requirements_[buffer_count_];
  if (AddBuffer(error_reporter, size, first_time_used, input_of_operators, 
      output_of_operators, operator_size, last_time_used) !=
      kTfLiteOk) {
    return kTfLiteError;
  }
  current->offline_offset = offline_offset;
  return kTfLiteOk;
}

bool TopologicalMemoryPlanner::DoesEntryOverlapInTime(
    const TopologicalMemoryPlanner::ListEntry* entry, const int first_time_used,
    const int last_time_used) const {
  const BufferRequirements* entry_requirements =
      &requirements_[entry->requirements_index];
  if (entry_requirements->first_time_used > last_time_used) {
    return false;
  }
  if (first_time_used > entry_requirements->last_time_used) {
    return false;
  }
  return true;
}

TopologicalMemoryPlanner::ListEntry*
TopologicalMemoryPlanner::NextSimultaneouslyActiveBuffer(
    const TopologicalMemoryPlanner::ListEntry* start, const int first_time_used,
    const int last_time_used) {
  ListEntry* result = nullptr;
  ListEntry* candidate_next_entry;
  if (start == nullptr) {
    candidate_next_entry = &buffers_sorted_by_offset_[first_entry_index_];
  } else {
    if (start->next_entry_index == -1) {
      return nullptr;
    }
    candidate_next_entry = &buffers_sorted_by_offset_[start->next_entry_index];
  }
  do {
    if (DoesEntryOverlapInTime(candidate_next_entry, first_time_used,
                               last_time_used)) {
      result = candidate_next_entry;
      break;
    }
    if (candidate_next_entry->next_entry_index == -1) {
      break;
    }
    candidate_next_entry =
        &buffers_sorted_by_offset_[candidate_next_entry->next_entry_index];
  } while (true);
  return result;
}

void TopologicalMemoryPlanner::CalculateOffsetsIfNeeded() {
  if (!need_to_calculate_offsets_ || (buffer_count_ == 0)) {
    return;
  }
  need_to_calculate_offsets_ = false;

  // Start off by ordering the buffers in descending order of size.
  // This helps find a more compact layout. Intuitively, you can think
  // about putting the large buffers in place first, and then the
  // smaller buffers can fit in the gaps, rather than fragmenting the
  // gaps with small buffers at the beginning. Add offline planned offsets
  // first in the list, since they have a predetermined offset.
  int idx_from_tail = buffer_count_;
  int idx_from_head = 0;
  for (int i = 0; i < buffer_count_; ++i) {
    if (requirements_[i].offline_offset == kOnlinePlannedBuffer) {
      idx_from_tail--;
      buffer_created_sorted_[idx_from_tail] = requirements_[i].first_time_used;
      buffer_ids_sorted_[idx_from_tail] = i;
      buffer_offsets_[i] = -1;
    } else {
      buffer_created_sorted_[idx_from_head] = requirements_[i].last_time_used;
      buffer_ids_sorted_[idx_from_head] = i;
      buffer_offsets_[i] = requirements_[i].offline_offset;
      idx_from_head++;
    }
  }

  // Do we need this? AddTensor have already gurateened that buffers
  // are added according to the operator's order
  // This sorting algorithm is naive, and may end up taking a very long time
  // with hundreds of buffers. Do not sort the offline planned offsets.
  //SortInPlace(&buffer_sizes_sorted_[idx_from_head],
  //                   &buffer_ids_sorted_[idx_from_head],
  //                   buffer_count_ - idx_from_head);

  // if buffer_count_ ==0, return at first line; So don't worry error
  // to access requirements_[0]
  int base_first_time_used = requirements_[0]->first_time_used;
  int i = 0;
  while (i < buffer_count_) {
    int base_first_time_used = requirements_[i]->first_time_used;
    while(current_first_time_used == base_first_time_used && i < buffer_count_) {
      i++;
      continue;
    }
    base_first_time_used = current_first_time_used;
  }

  // Initialize the first entry to the first buffer in
  // buffer_ids_sorted_.
  //   - If there are no offline planned offsets, the largest buffer will be
  //     first, and the buffers will be handled in size order.
  //   - If offline offsets are present, these will be handled first in order
  //     for the greedy algorithm to utilized gaps in the offline plan.
  first_entry_index_ = 0;
  next_free_entry_ = 1;
  ListEntry* first_entry = &buffers_sorted_by_offset_[first_entry_index_];
  first_entry->next_entry_index = -1;  // to mark the entry as end of list
  int buffer_id = buffer_ids_sorted_[0];
  first_entry->requirements_index = buffer_id;
  if (requirements_[buffer_id].offline_offset == kOnlinePlannedBuffer) {
    buffer_offsets_[buffer_id] = 0;
  }
  first_entry->offset = buffer_offsets_[buffer_id];

  // Work through the rest of the buffers to find a good gap to place each one.
  for (int i = 1; i < buffer_count_; ++i) {
    // The id is the order the buffer was originally added by the client.
    buffer_id = buffer_ids_sorted_[i];
    // Look at what size and time range the buffer needs to be active.
    BufferRequirements* wanted_requirements = &requirements_[buffer_id];
    const int wanted_size = wanted_requirements->size;
    const int wanted_first_time_used = wanted_requirements->first_time_used;
    const int wanted_last_time_used = wanted_requirements->last_time_used;

    // Find the first buffer that's active in our time range. All placed
    // buffers are stored in the order of their starting position in the arena
    // so that it's easy to find the next buffer in memory, and so the gap.
    // The candidate_entry variable holds the buffer that we're considering
    // placing the current buffer after.

    int candidate_offset = 0;
    // Loop through the offset-ordered list of buffers, looking for gaps.
    if (wanted_requirements->offline_offset == kOnlinePlannedBuffer) {
      ListEntry* prior_entry = nullptr;
      while (true) {
        // Find out what the next active buffer is.
        ListEntry* next_entry = NextSimultaneouslyActiveBuffer(
            prior_entry, wanted_first_time_used, wanted_last_time_used);

        if (prior_entry) {
          BufferRequirements* candidate_requirements =
              &requirements_[prior_entry->requirements_index];
          const int prior_entry_offset =
              prior_entry->offset + candidate_requirements->size;
          if (prior_entry_offset > candidate_offset) {
            candidate_offset = prior_entry_offset;
          }
        }
        if (next_entry == nullptr) {
          // We're at the end of the list, so we can always append the buffer
          // here.
          break;
        }
        // Find out how much space there is between us and the next buffer.
        const int gap = next_entry->offset - candidate_offset;
        if (gap >= wanted_size) {
          // This entry has a big enough gap between it and the next, so
          // use it!
          break;
        }
        // The gap wasn't big enough, so move on to another candidate.
        prior_entry = next_entry;
      }
    } else {
      // Offline planned offset are to be considered constant
      candidate_offset = wanted_requirements->offline_offset;
    }
    // At this point, we've either found a gap (possibly at the end of the
    // list) and want to place the buffer there, or there are no other active
    // buffers in this time range and so we can put it at offset zero.
    // Record the buffer's offset in our plan.
    buffer_offsets_[buffer_id] = candidate_offset;
    // Add the newly-placed buffer to our offset-ordered list, so that
    // subsequent passes can fit in their buffers around it.
    ListEntry* new_entry = &buffers_sorted_by_offset_[next_free_entry_];
    new_entry->offset = candidate_offset;
    new_entry->requirements_index = buffer_id;
    const int new_entry_index = next_free_entry_;
    ++next_free_entry_;

    if (first_entry->offset > candidate_offset) {
      // The new entry offset is smaller than the first entry offset =>
      // replace the first entry
      first_entry = new_entry;
      first_entry->next_entry_index = first_entry_index_;
      first_entry_index_ = new_entry_index;
    } else {
      ListEntry* current_entry = first_entry;
      // Make sure that we insert the buffer at the correct place in the
      // buffer-offset-ordered list
      while (true) {
        const int next_entry_index = current_entry->next_entry_index;
        if (next_entry_index == -1) {
          // We're at the end of the list, so just add the new entry here.
          current_entry->next_entry_index = new_entry_index;
          new_entry->next_entry_index = -1;
          break;
        }
        // not at the end of the list -> take a look at next entry
        ListEntry* next_entry = &buffers_sorted_by_offset_[next_entry_index];
        if (next_entry->offset > candidate_offset) {
          // We're at the right spot to do an insertion and retain the sorting
          // order, so place the new entry here.
          new_entry->next_entry_index = current_entry->next_entry_index;
          current_entry->next_entry_index = new_entry_index;
          break;
        }
        current_entry = next_entry;
      }
    }
  }
}

size_t TopologicalMemoryPlanner::GetMaximumMemorySize() {
  CalculateOffsetsIfNeeded();
  if (buffer_count_ == 0) {
    return 0;
  }
  ListEntry* entry = &buffers_sorted_by_offset_[first_entry_index_];
  size_t max_size = 0;
  while (entry) {
    BufferRequirements* requirements =
        &requirements_[entry->requirements_index];
    const size_t current_size = entry->offset + requirements->size;
    if (current_size > max_size) {
      max_size = current_size;
    }
    if (entry->next_entry_index == -1) {
      break;
    }
    entry = &buffers_sorted_by_offset_[entry->next_entry_index];
  }
  return max_size;
}

void TopologicalMemoryPlanner::PrintMemoryPlan() {
  CalculateOffsetsIfNeeded();

  for (int i = 0; i < buffer_count_; ++i) {
    MicroPrintf("%c (id=%d): size=%d, offset=%d, first_used=%d last_used=%d",
                GetOrdinalCharacter(i), i, requirements_[i].size,
                buffer_offsets_[i], requirements_[i].first_time_used,
                requirements_[i].last_time_used);
  }

  constexpr int kLineWidth = 80;
  int max_size = kLineWidth;
  int max_time = 0;
  for (int i = 0; i < buffer_count_; ++i) {
    BufferRequirements* requirements = &requirements_[i];
    const int offset = buffer_offsets_[i];
    const int last_time_used = requirements->last_time_used;
    const int size = offset + requirements->size;
    if (size > max_size) {
      max_size = size;
    }
    if (last_time_used > max_time) {
      max_time = last_time_used;
    }
  }

  char line[kLineWidth + 1];
  for (int t = 0; t <= max_time; ++t) {
    for (int c = 0; c < kLineWidth; ++c) {
      line[c] = '.';
    }
    int memory_use = 0;
    for (int i = 0; i < buffer_count_; ++i) {
      BufferRequirements* requirements = &requirements_[i];
      if ((t < requirements->first_time_used) ||
          (t > requirements->last_time_used)) {
        continue;
      }
      const int offset = buffer_offsets_[i];
      if (offset == -1) {
        continue;
      }
      const int size = requirements->size;
      memory_use += size;
      const int line_start = (offset * kLineWidth) / max_size;
      const int line_end = ((offset + size) * kLineWidth) / max_size;
      for (int n = line_start; n < line_end; ++n) {
        if (line[n] == '.') {
          line[n] = GetOrdinalCharacter(i);
        } else {
          line[n] = '!';
        }
      }
    }
    line[kLineWidth] = 0;

    MicroPrintf("%s%d: %s (%dk)", t < 10 ? " " : "", t, (const char*)line,
                (memory_use + 1023) / 1024);
  }
}

int TopologicalMemoryPlanner::GetBufferCount() { return buffer_count_; }

TfLiteStatus TopologicalMemoryPlanner::GetOffsetForBuffer(
    tflite::ErrorReporter* error_reporter, int buffer_index, int* offset) {
  CalculateOffsetsIfNeeded();
  if ((buffer_index < 0) || (buffer_index >= buffer_count_)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "buffer index %d is outside range 0 to %d",
                         buffer_index, buffer_count_);
    return kTfLiteError;
  }
  *offset = buffer_offsets_[buffer_index];
  return kTfLiteOk;
}

bool TopologicalMemoryPlanner::DoAnyBuffersOverlap(ErrorReporter* error_reporter) {
  CalculateOffsetsIfNeeded();
  bool were_overlaps_found = false;
  for (int i = 0; i < buffer_count_; ++i) {
    BufferRequirements* a_requirements = &requirements_[i];
    const int a_start_offset = buffer_offsets_[i];
    const int a_first_time_used = a_requirements->first_time_used;
    const int a_last_time_used = a_requirements->last_time_used;
    const int a_end_offset = a_start_offset + a_requirements->size;
    for (int j = 0; j < buffer_count_; ++j) {
      if (i == j) {
        continue;
      }
      BufferRequirements* b_requirements = &requirements_[j];
      const int b_start_offset = buffer_offsets_[j];
      const int b_first_time_used = b_requirements->first_time_used;
      const int b_last_time_used = b_requirements->last_time_used;
      const int b_end_offset = b_start_offset + b_requirements->size;
      if ((a_first_time_used > b_last_time_used) ||
          (b_first_time_used > a_last_time_used)) {
        // Buffers don't overlap in time.
        continue;
      }
      if ((a_start_offset >= b_end_offset) ||
          (b_start_offset >= a_end_offset)) {
        // No overlap in memory.
        continue;
      }
      were_overlaps_found = true;
      TF_LITE_REPORT_ERROR(
          error_reporter, "Overlap: %d (%d=>%d, %d->%d) vs %d (%d=>%d, %d->%d)",
          i, a_first_time_used, a_last_time_used, a_start_offset, a_end_offset,
          j, b_first_time_used, b_last_time_used, b_start_offset, b_end_offset);
    }
  }
  return were_overlaps_found;
}

}  // namespace tflite
