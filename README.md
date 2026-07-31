# Olive Engine - Plate-Solving Engine in Rust

A high performance astronomical plate-solving engine built on top of [`olive-solve`](https://github.com/oakamil/olive-solve).

## Features

* Core plate-solving engine that is optimized for speed
* Implementation of the LX200 protocol for control through SkySafari (Plus/Pro) or Stellarium+ mobile applications
* Optional ultra-fast camera pipeline option that uses the Raspberry Pi's ISP, enabling a solve rate of up to 60 Hz
* Configurable extraction engine, integrating with both the `olive-solve` and `cedar-detect` extractors

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
* `--extractor-type <EXTRACTOR_TYPE>`: (Optional) Extractor type to use (`approximate`, `fast`, `cedar`). Default is `approximate`.

### Web server

`olive-engine` runs a lightweight web server on port 8080. The server home page shows performance statistics. It also provides a live view of the camera feed that can be used to dial in the focus of the camera's lens. When the live view is active the plate-solving engine is paused.

## Roadmap

Additional features are planned for this solving engine, including:

1. Pre-built image for the Raspberry Pi platform
2. Better UVC camera support (current implementation is a work in progress)
3. IMU integration, initially targeting BNO085 and BMI160 sensors
4. Bluetooth control

## FAQ

1\. Which extractor type and mode should I use?

`cedar` provides the fastest performance, but its use is [restricted](https://github.com/smroid/cedar-detect/blob/main/LICENSE.md). `approximate` sacrifices a bit of accuracy for performance, and is recommended on lower end hardware. `fast` utilizes a robust algorithm for finding centroids. The background subtraction modes apply to `approximate` and `fast`, with `block` being the slowest but applying semi-local background processing. `global` is generally slightly faster than `line`, but `line` can better deal with certain types of image artifacts.

2\. What's ISP mode and why should I use it?

ISP mode utilizes the Raspberry Pi's built-in image signal processor to perform debayering and downsampling of images. It has 2 benefits. The available frame rate increases from ~40fps to the typical sensor maximum of 60fps, and the post-processing time drops significantly since the resulting format requires a memory copy only.

Note that ISP mode requires custom tuning files to turn off most image processing in the ISP.

3\. What hardware should I use?

The recommended hardware configuration is as follows:

* Raspberry Pi Zero 2W
* Innomaker IMX462 color module
* 25mm M12 lens
* 32GB or larger microSD card
* (Optional) Heat sinks
* (Optional) [3D printed case](https://www.printables.com/model/1648425-cedar-zero-case-v2)

4\. Can I use this engine in my own project?

Yes! Part of the goal here is to provide a core engine that can be used in other projects. Rust and Python entry points are provided in order to make it easier to integrate the engine into other systems.

There are some restrictions though:
* The `cedar` extractor type uses `cedar-detect` which has a restricted license. Commercial projects that intend to use that path will need to negotiate a license with Clear Skies Astro.
* The other extractor types and the core solver are subject to the licensing of the `olive-solve` project, which currently limits commercial use to small businesses.

## License

This project is licensed under the Apache License, Version 2.0.

See LICENSE.md for full details.

### Third-Party Licenses

While `olive-engine` itself is licensed under the Apache License 2.0, it integrates with and depends on several external projects (including but not limited to `olive-solve` and `cedar-detect`). Each of these third-party projects is governed by its own respective licensing terms. Users are responsible for reviewing and complying with the individual licenses of any integrated components, tools, or dependencies.

## Disclaimer

All product names, trademarks and registered trademarks are property of their respective owners. All company, product and service names used in this website are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.
