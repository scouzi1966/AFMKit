/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include <algorithm>

#include "mlx/c/io.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/io.h"

extern "C" int
mlx_load_reader(mlx_array* res, mlx_io_reader in_stream, const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::load(mlx_io_reader_get_(in_stream), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_load(mlx_array* res, const char* file, const mlx_stream s) {
  try {
    mlx_array_set_(
        *res, mlx::core::load(std::string(file), mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_load_safetensors_reader(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    mlx_io_reader in_stream,
    const mlx_stream s) {
  try {
    {
      auto [tpl_0, tpl_1] = mlx::core::load_safetensors(
          mlx_io_reader_get_(in_stream), mlx_stream_get_(s));
      mlx_map_string_to_array_set_(*res_0, tpl_0);
      mlx_map_string_to_string_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_load_safetensors(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    const char* file,
    const mlx_stream s) {
  try {
    {
      auto [tpl_0, tpl_1] =
          mlx::core::load_safetensors(std::string(file), mlx_stream_get_(s));
      mlx_map_string_to_array_set_(*res_0, tpl_0);
      mlx_map_string_to_string_set_(*res_1, tpl_1);
    };
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_load_gguf(
    mlx_map_string_to_array* arrays,
    mlx_map_string_to_array* array_metadata,
    mlx_map_string_to_string* string_metadata,
    const char* file,
    const mlx_stream s) {
  try {
    auto [loaded_arrays, loaded_metadata] =
        mlx::core::load_gguf(std::string(file), mlx_stream_get_(s));
    std::unordered_map<std::string, mlx::core::array> metadata_arrays;
    std::unordered_map<std::string, std::string> metadata_strings;
    for (auto& [key, value] : loaded_metadata) {
      if (auto* array = std::get_if<mlx::core::array>(&value)) {
        metadata_arrays.emplace(key, std::move(*array));
      } else if (auto* string = std::get_if<std::string>(&value)) {
        metadata_strings.emplace(key, *string);
      } else if (auto* strings = std::get_if<std::vector<std::string>>(&value)) {
        // Preserve string arrays losslessly without adding a new C map ABI.
        size_t encoded_size = 2 + (strings->empty() ? 0 : strings->size() - 1);
        for (const auto& string : *strings) {
          encoded_size += 2 + string.size();
          encoded_size += std::count_if(
              string.begin(), string.end(), [](char character) {
                return character == '\\' || character == '\"';
              });
        }
        std::string encoded = "[";
        encoded.reserve(encoded_size);
        for (size_t index = 0; index < strings->size(); ++index) {
          if (index) encoded += ",";
          encoded += "\"";
          for (char character : strings->at(index)) {
            if (character == '\\' || character == '\"') encoded += '\\';
            encoded += character;
          }
          encoded += "\"";
        }
        encoded += "]";
        metadata_strings.emplace(key, std::move(encoded));
      }
    }
    mlx_map_string_to_array_set_(*arrays, loaded_arrays);
    mlx_map_string_to_array_set_(*array_metadata, metadata_arrays);
    mlx_map_string_to_string_set_(*string_metadata, metadata_strings);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_save_writer(mlx_io_writer out_stream, const mlx_array a) {
  try {
    mlx::core::save(mlx_io_writer_get_(out_stream), mlx_array_get_(a));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_save(const char* file, const mlx_array a) {
  try {
    mlx::core::save(std::string(file), mlx_array_get_(a));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_save_safetensors_writer(
    mlx_io_writer in_stream,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata) {
  try {
    mlx::core::save_safetensors(
        mlx_io_writer_get_(in_stream),
        mlx_map_string_to_array_get_(param),
        mlx_map_string_to_string_get_(metadata));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_save_safetensors(
    const char* file,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata) {
  try {
    mlx::core::save_safetensors(
        std::string(file),
        mlx_map_string_to_array_get_(param),
        mlx_map_string_to_string_get_(metadata));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
