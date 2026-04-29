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

## Usage

### olive-engine

The core plate-solving engine daemon. It controls the camera pipeline, extracts stars, and solves the astronomical plates.

```Bash
cargo run --release -- --database-path ~/data/default_database.npz
```
* `--min-exposure <MIN_EXPOSURE>`: (Optional) Minimum exposure time in microseconds. Default is 16670.
* `--max-exposure <MAX_EXPOSURE>`: (Optional) Maximum exposure time in microseconds. Default is 1000000.
* `--gain <GAIN>`: (Optional) Analog gain to apply to the camera.
* `--target-centroids <TARGET_CENTROIDS>`: (Optional) Target number of centroids for auto-exposure. Default is 20.
* `--database-path <DATABASE_PATH>`: (Optional) Path to the Tetra3 database NPZ file. Default is `$HOME/data/default_database.npz`.
* `--isp`: (Optional) Enable ISP mode configuration for the ultra-fast camera pipeline.
* `--uvc`: (Optional) Use UVC camera instead of libcamera.
* `--bg-mode <BG_MODE>`: (Optional) Background subtraction mode for extraction (`line`, `global`, `block`). Default is `line`.
* `--extractor-type <EXTRACTOR_TYPE>`: (Optional) Extractor type to use (`sequential`, `fast`, `cedar`). Default is `sequential`.
* `--rayon-threads <RAYON_THREADS>`: (Optional) Rayon threads configuration. Default is 2.

## Roadmap

Additional features are planned for this solving engine, including:

1. Better UVC camera support (current implementation is a work in progress)
2. IMU integration, initially targeting BNO085 and BMI160 sensors
3. Bluetooth control

## FAQ

1. Which extractor type and mode should I use?

`cedar` provides the fastest performance, but its use is [restricted](https://github.com/smroid/cedar-detect/blob/main/LICENSE.md). `sequential` sacrifices a bit of accuracy for performance, and is recommended on lower end hardware. `fast` utilizes rayon multi-threading which can hold up CPU threads on lower end hardware. The background subtraction modes apply to `sequential` and `fast`, with `block` being the slowest but applying semi-local background processing. `global` is generally slightly faster than `line`, but `line` can better deal with certain types of image artifacts.

2. What's ISP mode and why should I use it?

ISP mode utilizes the Raspberry Pi's built-in image signal processor to perform debayering and downsampling of images. It has 2 benefits. The available frame rate increases from ~40fps to the typical sensor maximum of 60fps, and the post-processing time drops significantly since the resulting format requires a memory copy only.

Note that ISP mode requires custom tuning files to turn off most image processing in the ISP.

3. What hardware should I use?

The recommended hardware configuration is as follows:

* Raspberry Pi Zero 2W
* Innomaker IMX462 color module
* 25mm M12 lens
* 32GB or larger microSD card
* (Optional) Heat sinks
* (Optional) [3D printed case](https://www.printables.com/model/1648425-cedar-zero-case-v2)

## License

This project is licensed under the Apache License, Version 2.0.

See LICENSE.md for full details.

### Third-Party Licenses

While `olive-engine` itself is licensed under the Apache License 2.0, it integrates with and depends on several external projects (including but not limited to `olive-solve` and `cedar-detect`). Each of these third-party projects is governed by its own respective licensing terms. Users are responsible for reviewing and complying with the individual licenses of any integrated components, tools, or dependencies.

## Disclaimer

All product names, trademarks and registered trademarks are property of their respective owners. All company, product and service names used in this website are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.
