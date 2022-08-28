use std::mem::MaybeUninit;
use nalgebra::{DMatrix, Dynamic, Matrix, Storage};

pub fn each_ref<T, const N: usize>(data: &[T; N]) -> [&T; N] {
    // Unlike in `map`, we don't need a guard here, as dropping a reference
    // is a noop.
    let mut out = [MaybeUninit::uninit(); N];
    for (src, dst) in data.iter().zip(&mut out) {
        dst.write(src);
    }

    // SAFETY: All elements of `dst` are properly initialized and
    // `MaybeUninit<T>` has the same layout as `T`, so this cast is valid.
    unsafe { (&mut out as *mut _ as *mut [&T; N]).read() }
}