//! # CUDA Standard Library
//!
//! The CUDA Standard Library provides a curated set of abstractions for writing performant, reliable, and
//! understandable GPU kernels using the Rustc NVVM backend.
//!
//! This library will build on non-nvptx targets or targets not using the nvvm backend. However, it will not
//! be usable, and it will throw linker errors if you attempt to use most of the functions in the library.
//! However, [`kernel`] automatically cfg-gates the function annotated for `nvptx64` or `nvptx`, therefore,
//! no "actual" functions from this crate should be used when compiling for a non-nvptx target.
//!
//! This crate cannot be used with the llvm ptx backend either, it heavily relies on external functions implicitly
//! defined by the nvvm backend, as well as internal attributes.
//!
//! # Structure
//!
//! This library tries to follow the structure of the Rust standard library to some degree, where
//! different concepts are separated into their own modules.
//!
//! # The Prelude
//!
//! In order to simplify imports, we provide a prelude module which contains GPU analogues to standard library
//! structures as well as common imports such as [`thread`].

#![feature(abi_ptx, stdsimd, concat_idents)]
#![no_std]

extern crate alloc;

pub mod float;
#[allow(warnings)]
pub mod intrinsics;

mod float_ext;

pub use cuda_std_macros::*;
pub use float::GpuFloat;
pub use float_ext::*;
