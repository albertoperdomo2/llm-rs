#![feature(portable_simd)]

use std::simd::{f32x8};
// f32x8 works on most CPUs but f32x16 would be better I think

pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // a is mxn, b is nxk and c is mxk
    // all matrices are stored row-major as flat vectors
    const SIMD_WIDTH: usize = 8; // process 8 f32s at once
    let mut vec = vec![0.0; m * k];

    // transpose B for more straight access
    let b_transposed = transpose(b, n, k);

    if n < SIMD_WIDTH {
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += a[i * n + l] * b_transposed[j * n + l];
                }
                vec[i * k + j] = sum;
            }
        }
        return vec;
    }

    for i in 0..m {
        for j in 0..k {
            let mut sum = f32x8::splat(0.0);
            let a_row = &a[i * n..(i + 1) * n];
            let b_row = &b_transposed[j * n..(j + 1) * n];

            // process 8 elements at once
            let chunks = n / SIMD_WIDTH;
            for chunk in 0..chunks {
                let start = chunk * SIMD_WIDTH;
                let a_vec = f32x8::from_slice(&a_row[start..start + SIMD_WIDTH]);
                let b_vec = f32x8::from_slice(&b_row[start..start + SIMD_WIDTH]);
                sum += a_vec * b_vec;
            }

            let sum_array: [f32; 8] = sum.into();
            let mut final_sum = sum_array.iter().sum::<f32>();

            // handle remaining elements
            for l in (chunks * SIMD_WIDTH)..n {
                final_sum += a_row[l] * b_row[l];
            }

            vec[i * k + j] = final_sum;
        }
    }

    vec
}

pub fn transpose(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..cols {
        for j in 0..rows {
            transposed[i * rows + j] = matrix[j * cols + i];
        }
    }

    transposed
}