# Olive Engine - Plate-Solving Engine in Rust

A robust, fast, and async-friendly Rust implementation for astronomical plate-solving.

## Features

* Core plate-solving engine that is optimized for speed
* Implementation of the LX200 protocol for control through SkySafari (Plus/Pro) or Stellarium+ mobile applications
* Optional ultra-fast camera pipeline option that uses the Raspberry Pi's ISP, enabling a solve rate of up to 60 Hz
* Configurable extraction engine, integrating with both the `olive-solve` and `cedar-detect` extractors

## Repository Structure

* **`src`**: The core application and library.
  * Device control and coordination logic (`engine.rs`, `camera.rs`, etc.).
  * Server endpoints for control and telemetry (`control_server.rs`, `telescope_server.rs`).
  * Python bindings for the engine (`python.rs`).
* **`build_python.sh`**: Script to build the Python extension module using `maturin`.

## Getting Started

### Prerequisites
* [Rust / Cargo](https://rustup.rs/)
* Python 3
* Maturin (for Python bindings)

### Building
To build the Rust crate:

```
cargo build --release
```

To build the Python wheel:

```
./build_python.sh
```

### Testing

Run the test suite:

```
cargo test --release
```

## Roadmap

Additional features are planned for this solving engine, including:

1. Better UVC camera support (current implementation is a work in progress)
2. IMU integration, initially targeting BNO085 and BMI160 sensors
3. Bluetooth control

## License

This project is licensed under the Apache License, Version 2.0.

See LICENSE.md for full details.

### Third-Party Licenses

While `olive-engine` itself is licensed under the Apache License 2.0, it integrates with and depends on several external projects (including but not limited to `olive-solve` and `cedar-detect`). Each of these third-party projects is governed by its own respective licensing terms. Users are responsible for reviewing and complying with the individual licenses of any integrated components, tools, or dependencies.

## Disclaimer

All product names, trademarks and registered trademarks are property of their respective owners. All company, product and service names used in this website are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.
