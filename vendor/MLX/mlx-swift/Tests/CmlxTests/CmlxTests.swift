// Copyright © 2024 Apple Inc.

//
//  Copyright © 2023 Apple. All rights reserved.
//

import Foundation
import XCTest

@testable import Cmlx

class CmlxTests: XCTestCase {

    func testSetDefaultStreamIsObservableThroughGet() {
        let device = mlx_device_new_type(MLX_GPU, 0)
        defer { mlx_device_free(device) }

        var original = mlx_stream_new()
        XCTAssertEqual(mlx_get_default_stream(&original, device), 0)
        defer {
            XCTAssertEqual(mlx_set_default_stream(original), 0)
            mlx_stream_free(original)
        }

        let replacement = mlx_stream_new_device(device)
        defer { mlx_stream_free(replacement) }
        XCTAssertEqual(mlx_set_default_stream(replacement), 0)

        var observed = mlx_stream_new()
        defer { mlx_stream_free(observed) }
        XCTAssertEqual(mlx_get_default_stream(&observed, device), 0)
        XCTAssertTrue(mlx_stream_equal(replacement, observed))
    }

    func testMinimal() throws {
        // smoke test making sure we can build, link & call C api
        //
        // note: there are convenience wrappers in MLX + the entire
        // wrapping of the API in swift

        var data: [Float] = [1, 2, 3, 4, 5, 6]
        var shape: [Int32] = [2, 3]

        let arr = mlx_array_new_data(&data, &shape, 2, MLX_FLOAT32)
        defer { mlx_array_free(arr) }

        var str = mlx_string_new()
        mlx_array_tostring(&str, arr)
        defer { mlx_string_free(str) }
        let description = String(cString: mlx_string_data(str))

        print(description)
    }

}
