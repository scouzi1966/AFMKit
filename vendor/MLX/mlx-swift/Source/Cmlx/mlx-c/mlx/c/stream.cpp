/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>
#include <map>
#include <mutex>

#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/stream.h"
#include "mlx/backend/metal/device.h"

namespace {

struct SwiftDefaultStreams {
  mlx::core::Stream get(mlx::core::Device device) {
    std::lock_guard lock(mutex);
    auto it = streams.find(device);
    if (it == streams.end()) {
      it = streams
               .emplace(
                   device, mlx::core::new_thread_unsafe_stream(device))
               .first;
    }
    return it->second;
  }

  void set(mlx::core::Stream stream) {
    std::lock_guard lock(mutex);
    streams.insert_or_assign(stream.device, stream);
  }

 private:
  std::map<mlx::core::Device, mlx::core::Stream> streams;
  std::mutex mutex;
};

SwiftDefaultStreams& swift_default_streams() {
  static SwiftDefaultStreams streams;
  return streams;
}

mlx::core::Stream swift_default_stream(mlx::core::Device device) {
  return swift_default_streams().get(device);
}

} // namespace

int mlx_stream_tostring(mlx_string* str_, mlx_stream stream) {
  try {
    std::ostringstream os;
    os << mlx_stream_get_(stream);
    std::string str = os.str();
    mlx_string_set_(*str_, str);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_stream mlx_stream_new(void) {
  return mlx_stream_new_();
}

extern "C" mlx_stream mlx_stream_new_device(mlx_device dev) {
  try {
    // Swift tasks can resume on a different thread after suspension. A
    // regular MLX stream is registered only on its creating thread, so a
    // Stream carried by TaskLocal state can otherwise become invalid after
    // an await. Use MLX's cross-thread stream for Swift-owned stream objects.
    return mlx_stream_new_(
        mlx::core::new_thread_unsafe_stream(mlx_device_get_(dev)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_stream_new_();
  }
}
extern "C" int mlx_stream_set(mlx_stream* stream, const mlx_stream src) {
  try {
    mlx_stream_set_(*stream, mlx_stream_get_(src));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_stream_free(mlx_stream stream) {
  try {
    mlx_stream_free_(stream);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" bool mlx_stream_equal(mlx_stream lhs, mlx_stream rhs) {
  return mlx_stream_get_(lhs) == mlx_stream_get_(rhs);
}
extern "C" int mlx_stream_get_device(mlx_device* dev, mlx_stream stream) {
  try {
    mlx_device_set_(*dev, mlx_stream_get_(stream).device);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
extern "C" int mlx_stream_get_index(int* index, mlx_stream stream) {
  try {
    *index = mlx_stream_get_(stream).index;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_synchronize(mlx_stream stream) {
  try {
    mlx::core::synchronize(mlx_stream_get_(stream));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_command_buffer_profile_since_report(
    mlx_command_buffer_profile* profile, mlx_stream stream) {
  try {
    auto& encoder = mlx::core::metal::get_command_encoder(
        mlx_stream_get_(stream));
    const auto result = encoder.command_buffer_profile_since_report();
    profile->buffers = result.buffers;
    profile->ops = result.ops;
    profile->bytes = result.bytes;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_default_stream(mlx_stream* stream, mlx_device dev) {
  try {
    mlx_stream_set_(*stream, swift_default_stream(mlx_device_get_(dev)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
extern "C" int mlx_set_default_stream(mlx_stream stream) {
  try {
    const auto& value = mlx_stream_get_(stream);
    // Preserve the MLX core contract for callers on this thread and update
    // the Swift cross-thread default observed by mlx_get_default_stream.
    mlx::core::set_default_stream(value);
    swift_default_streams().set(value);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_stream mlx_default_cpu_stream_new(void) {
  try {
    return mlx_stream_new_(swift_default_stream(mlx::core::Device::cpu));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_stream_new_();
  }
}
extern "C" mlx_stream mlx_default_gpu_stream_new(void) {
  try {
    return mlx_stream_new_(swift_default_stream(mlx::core::Device::gpu));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_stream_new_();
  }
}
