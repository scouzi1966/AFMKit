// Copyright © 2023-2024 Apple Inc.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::gpu {

namespace {

bool profile_primitives() {
  static const bool enabled = [] {
    const char* value = std::getenv("MLX_PROFILE_PRIMITIVES");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

std::mutex primitive_profile_mutex;
std::unordered_map<int, std::unordered_map<std::string, uint64_t>>
    primitive_profiles;

std::string shape_string(const Shape& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      result += ",";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

void record_primitive(Stream stream, const array& arr) {
  if (!profile_primitives()) {
    return;
  }
  std::string key = arr.primitive().name();
  if (key == "Transpose" && !arr.inputs().empty()) {
    key += shape_string(arr.inputs().front().shape());
    key += "->";
    key += shape_string(arr.shape());
  }
  std::lock_guard<std::mutex> lock(primitive_profile_mutex);
  primitive_profiles[stream.index][key]++;
}

void report_primitives(Stream stream) {
  if (!profile_primitives()) {
    return;
  }

  std::vector<std::pair<std::string, uint64_t>> counts;
  {
    std::lock_guard<std::mutex> lock(primitive_profile_mutex);
    auto profile = primitive_profiles.find(stream.index);
    if (profile == primitive_profiles.end() || profile->second.empty()) {
      return;
    }
    counts.assign(profile->second.begin(), profile->second.end());
    primitive_profiles.erase(profile);
  }
  std::sort(
      counts.begin(),
      counts.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.second == rhs.second ? lhs.first < rhs.first
                                        : lhs.second > rhs.second;
      });

  uint64_t total = 0;
  std::fprintf(stderr, "[MLXPrimitives] stream=%d", stream.index);
  for (const auto& [name, count] : counts) {
    total += count;
    std::fprintf(
        stderr,
        " %s=%llu",
        name.c_str(),
        static_cast<unsigned long long>(count));
  }
  std::fprintf(
      stderr, " total=%llu\n", static_cast<unsigned long long>(total));
}

} // namespace

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_sets());
}

void new_thread_unsafe_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_global_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_sets());
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  record_primitive(s, arr);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    encoder.commit([s, buffers = std::move(buffers)]() {
      scheduler::notify_task_completion(s);
    });
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {});
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  encoder.commit();
}

void synchronize(Stream s) {
  auto& encoder = metal::get_command_encoder(s);
  encoder.synchronize();
  report_primitives(s);
  static const bool profile_command_buffers = [] {
    const char* value = std::getenv("MLX_PROFILE_COMMAND_BUFFERS");
    return value != nullptr && std::string(value) == "1";
  }();
  if (profile_command_buffers) {
    const auto profile = encoder.command_buffer_profile_since_report();
    std::fprintf(
        stderr,
        "[MLXCommandBuffers] stream=%d buffers=%llu ops=%llu bytes=%llu\n",
        s.index,
        static_cast<unsigned long long>(profile.buffers),
        static_cast<unsigned long long>(profile.ops),
        static_cast<unsigned long long>(profile.bytes));
  }
}

void clear_streams() {
  metal::get_command_encoders().clear();
  if (is_main_thread()) {
    metal::get_global_command_encoders().clear();
  }
}

} // namespace mlx::core::gpu
