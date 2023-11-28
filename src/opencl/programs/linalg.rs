use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::{build, TILE_SIZE, WG_SIZE};

#[memoize]
pub fn pad_matrices(c_type: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void pad_matrices(
                ulong2 const stride_in,
                ulong2 const stride_out,
                __global const {c_type}* restrict input,
                __global {c_type}* restrict output)
        {{
            const uint w = get_global_id(0);    // matrix number
            const uint x = get_global_id(1);    // row index
            const uint y = get_global_id(2);    // column index

            const ulong offset_in = (w * stride_in.x) + (x * stride_in.y) + y;

            const ulong offset_out = (w * stride_out.x) + (x * stride_out.y) + y;

            output[offset_out] = input[offset_in];
        }}
    "#
    );

    build(&src)
}

#[memoize]
pub fn matmul(c_type: &'static str) -> Result<Program, Error> {
    debug_assert_eq!(TILE_SIZE * TILE_SIZE, WG_SIZE);

    let src = format!(
        r#"
        __kernel void matmul(
                ulong4 const dims,
                ulong const reduce_tiles,
                __global const {c_type}* restrict left,
                __global const {c_type}* restrict right,
                __global {c_type}* restrict output)
        {{
            // x := output axis 0
            // y := reduce axis
            // z := output axis 1
            // w := matrix number

            const ulong w = get_global_id(0);
            const ulong x_tile = get_global_id(1);
            const ulong z_tile = get_global_id(2);

            const ulong x_offset = x_tile * {TILE_SIZE};
            const ulong z_offset = z_tile * {TILE_SIZE};
            const ulong left_offset = w * dims.x * dims.y;
            const ulong right_offset = w * dims.y * dims.z;

            {c_type} tile[{TILE_SIZE}][{TILE_SIZE}] = {{ 0 }};

            // initialize the local cache for the left and right tiles to zero
            {c_type} left_tile[{TILE_SIZE}][{TILE_SIZE}] = {{ 0 }};
            {c_type} right_tile[{TILE_SIZE}][{TILE_SIZE}] = {{ 0 }};

            // for each tile on the y axis
            for (ulong y_tile = 0; y_tile < reduce_tiles; y_tile++) {{
                const ulong y_offset = y_tile * {TILE_SIZE};

                // read the left and right tiles into the local cache
                #pragma unroll
                for (uint i = 0; i < {TILE_SIZE}; i++) {{
                    #pragma unroll
                    for (uint j = 0; j < {TILE_SIZE}; j++) {{
                        ulong offset_l = left_offset + ((x_offset + i) * dims.y) + (y_offset + j);
                        left_tile[i][j] = left[offset_l];

                        ulong offset_r = right_offset + ((y_offset + i) * dims.z) + (z_offset + j);
                        right_tile[i][j] = right[offset_r];
                    }}
                }}

                // tile += left tile @ right tile
                #pragma unroll
                for (uint i = 0; i < {TILE_SIZE}; i++) {{
                    #pragma unroll
                    for (uint j = 0; j < {TILE_SIZE}; j++) {{
                        #pragma unroll
                        for (uint k = 0; k < {TILE_SIZE}; k++) {{
                            tile[i][k] += left_tile[i][j] * right_tile[j][k];
                        }}
                    }}
                }}
            }}

            // write tile to output
            ulong tile_offset = (w * dims.x * dims.z) + (x_offset * dims.z) + z_offset;

            #pragma unroll
            for (uint i = 0; i < {TILE_SIZE}; i++) {{
                #pragma unroll
                for (uint j = 0; j < {TILE_SIZE}; j++) {{
                    if ((x_offset + i) < dims.x && (z_offset + j) < dims.z) {{
                        ulong offset = tile_offset + (i * dims.z) + j;
                        output[offset] = tile[i][j];
                    }}
                }}
            }}
        }}
        "#,
    );

    build(&src)
}
