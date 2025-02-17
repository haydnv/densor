# ha-ndarray
An n-dimensional array for Rust, with [OpenCL](https://www.khronos.org/opencl/) hardware acceleration
implemented using the [ocl](https://github.com/cogciprocate/ocl) crate.

Use the `opencl` feature flag to enable OpenCL support.

OpenCL is a trademark of Apple Inc. used by permission by the Khronos Group. For more information on OpenCL in general, see:

 - [A Gentle Introduction to OpenCL](https://freecontent.manning.com/wp-content/uploads/a-gentle-introduction-to-opencl.pdf) by Matthew Scarpino

 - [The OpenCL C Programming Language](https://registry.khronos.org/OpenCL/specs/2.2/html/OpenCL_C.html) published by the Khronos Group

 - This excellent overview of OpenCL kernel programming & optimization: https://www.nersc.gov/assets/pubs_presos/MattsonTutorialSC14.pdf

 - A benchmarking tool available for comparing numpy, ndarray and ha-ndarray is available in the `benchmark` branch and can be built with `cargo run --bin benchmark --features benchmark` see [README.md](./benchmark/README.md) for more information.
