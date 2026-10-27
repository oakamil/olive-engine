# olive-engine

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE.md)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](Cargo.toml)

High-performance astronomical plate-solving engine and telescope server daemon written in Rust with Python bindings. Built on top of [`olive-solve`](https://github.com/oakamil/olive-solve) and [`tetra3`](https://github.com/esa/tetra3).

---

## Features

- **Real-Time Astrometry**: Up to 60 Hz attitude determination and celestial tracking.
- **Telescope Protocol Servers**:
  - **Meade LX200**: High-speed TCP server on port 4030 compatible with SkySafari (Plus/Pro), Stellarium+, and INDI.
  - **ASCOM Alpaca**: Native HTTP REST API on port 11111 and UDP broadcast discovery on port 32227 compatible with N.I.N.A., Ekos/KStars, and SharpCap.
- **Embedded Web Control & Diagnostics**: Lightweight web interface on port 8080 with real-time attitude telemetry (RA, Dec, Roll), solve rates, Half-Flux Radius (HFR) lens focus metric, and multipart MJPEG live camera stream.
- **Dual Camera Backends**:
  - **Raspberry Pi libcamera**: Hardware ISP debayering and downsampling for zero-copy high-throughput capture.
  - **USB UVC Cameras**: Standard USB camera support with configurable framerate, MJPEG decoding, and multi-frame rolling exposure accumulation.
- **Inertial Measurement Unit (IMU) Sensor Fusion**: Fused attitude tracking using Bosch BMI160 or CEVA/Hillcrest BNO085 sensors for dead-reckoning and instant re-acquisition during slews.
- **Autonomous Adaptive Regulation**: Dynamic gain, exposure, and stacking controller that automatically adapts to changing sky brightness and cloud cover.
- **Configurable Centroid Extractors**: Integration with `approximate`, `fast`, and `cedar-detect` extractors with multiple background subtraction modes (`line`, `global`, `block`).
- **Python Bindings**: Native PyO3 extension (`olive_engine`) for seamless integration into custom Python astronomy pipelines.
- **Hardware Tuning Utilities**: Dedicated companion binaries (`uvc_tuner` and `uvc_web_tuner`) for benchmarking and dialing in sensor exposure, gain, and gamma.

---

## Network Services

`olive-engine` exposes the following network services:

| Port | Protocol | Service | Compatible Clients / Usage |
|---|---|---|---|
| `8080` | HTTP | Web UI & MJPEG Stream | Web browser (lens focusing, manual controls, telemetry) |
| `4030` | TCP | Meade LX200 Protocol | SkySafari Plus/Pro, Stellarium+, INDI |
| `11111` | HTTP | ASCOM Alpaca API | N.I.N.A., Ekos, SharpCap, ASCOM Alpaca clients |
| `32227` | UDP | ASCOM Discovery | Automated ASCOM Alpaca client discovery |

---

## Getting Started

### Prerequisites

- [Rust & Cargo](https://rustup.rs/) (1.85+ recommended, 2024 edition)
- Python 3.9+ & [Maturin](https://github.com/PyO3/maturin) (for Python extension)
- Tetra3 star database file (default: `$HOME/data/default_database.npz`)
- Linux / Raspberry Pi: `libcamera-dev` (if using Raspberry Pi camera)
- macOS / Linux: `libuvc` (vendored automatically via Cargo)

### Building

Build release binaries:
```bash
cargo build --release
```

Build and install the Python wheel:
```bash
./build_python.sh
```

### Running Tests

Run the test suite:
```bash
# On Linux:
cargo test --release

# On macOS (framework path required for PyO3 dynamic linking):
DYLD_FALLBACK_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/Developer/Library/Frameworks cargo test --release
```

---

## Usage

### Running the Daemon

```bash
cargo run --release -- --database-path ~/data/default_database.npz
```

### Command-Line Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--database-path <PATH>` | String | `$HOME/data/default_database.npz` | Path to Tetra3 star catalog (`.npz`) |
| `--isp` | Flag | `false` | Enable Raspberry Pi ISP hardware pipeline (libcamera only) |
| `--uvc` | Flag | `false` | Use UVC camera backend instead of libcamera |
| `--uvc-fps <INT>` | Int | `30` | Target framerate for UVC camera |
| `--uvc-mjpeg` | Flag | `false` | Enable MJPEG capture format for UVC |
| `--gain <FLOAT>` | Float | Auto | Fixed analog gain override (disables AGC) |
| `--target-centroids <INT>` | Int | `20` | Target star centroid count for auto-exposure |
| `--bg-mode <MODE>` | Enum | `line` | Background subtraction mode (`line`, `global`, `block`) |
| `--extractor-type <TYPE>` | Enum | `approximate` | Star extractor backend (`approximate`, `fast`, `cedar`) |
| `--imu <TYPE>` | Enum | `none` | IMU sensor fusion backend (`none`, `bno085`, `bmi160`) |
| `--no-downsample` | Flag | `false` | Disable automatic 2x2 image downsampling |
| `--auto-mode <BOOL>` | Bool | `true` | Automated gain/exposure/stacking regulation |

### Auxiliary Camera Tuning Tools

`olive-engine` includes companion binaries for testing and tuning UVC USB cameras:

- **CLI Tuner (`uvc_tuner`)**: Benchmarks camera control combinations (gamma, contrast, gain) against star detection rate:
  ```bash
  cargo run --release --bin uvc_tuner [-- --gain <GAIN>]
  ```
- **Web Tuner (`uvc_web_tuner`)**: Interactive browser-based UI for live-adjusting UVC camera controls and evaluating focus/detection:
  ```bash
  cargo run --release --bin uvc_web_tuner
  ```

---

## Python Quickstart

The Python extension module (`olive_engine`) exposes the core engine with real-time telemetry callbacks:

```python
import olive_engine

def on_solve(telemetry):
    if telemetry.get("ra") is not None:
        print(
            f"RA={telemetry['ra']:.3f}° Dec={telemetry['dec']:.3f}° "
            f"Rate={telemetry.get('solve_rate_hz', 0.0):.1f}Hz "
            f"HFR={telemetry.get('hfr', 0.0):.2f}px"
        )

# Start engine daemon (blocks until interrupted)
olive_engine.start_engine(
    isp=True,
    target_centroids=25,
    callback=on_solve
)
```

---

## Hardware Recommendations

- **Single Board Computer**: Raspberry Pi Zero 2 W, Raspberry Pi 4B, or Raspberry Pi 5.
- **Camera Sensor**: InnoMaker IMX462 or Arducam IMX462 (ultra-low-light sensitive sensor).
- **Optics**: 25mm M12 lens (~12° × 7° field of view).
- **Storage**: 32 GB or larger high-endurance microSD card.
- **IMU Sensor (Optional)**: Adafruit BNO085 or Bosch BMI160 connected via I2C/SPI.
- **Enclosure**: [3D-printed Cedar Zero Case v2](https://www.printables.com/model/1648425-cedar-zero-case-v2).

---

## FAQ

#### 1. Which extractor type and background mode should I use?
- `cedar`: Maximum performance and speed, but its use is subject to [cedar-detect license restrictions](https://github.com/smroid/cedar-detect).
- `approximate`: Recommended for resource-constrained platforms (e.g. Raspberry Pi Zero 2W).
- `fast`: High-precision centroiding algorithm for challenging optical setups.
- Background modes (`line`, `global`, `block`): `line` is the default and effectively mitigates rolling-shutter CMOS gradients. `global` is slightly faster. `block` applies local background modeling at higher CPU cost.

#### 2. What is ISP mode and why use it?
ISP mode uses the Raspberry Pi's hardware Image Signal Processor to handle debayering and downsampling in silicon. This increases throughput from ~40 FPS to the sensor maximum of 60 FPS and reduces frame post-processing time to a single memory copy. Custom camera tuning JSON files for libcamera ISP mode are provided in `data/pisp/` and `data/vc4/`.

---

## Roadmap

1. Pre-built Raspberry Pi OS disk images for turnkey setup.
2. Bluetooth Low Energy (BLE) telescope protocol support.
3. Expanded support for native motorized focuser drivers.

---

## License & Attribution

This project is licensed under the [GNU General Public License v3.0 or later (GPL-3.0-or-later)](LICENSE.md).

### Third-Party Licenses

`olive-engine` integrates with and relies on several upstream projects:
- [`olive-solve`](https://github.com/oakamil/olive-solve): Core astrometric plate solver.
- [`tetra3`](https://github.com/esa/tetra3): Star identification algorithm developed by the European Space Agency (ESA).
- [`cedar-detect`](https://github.com/smroid/cedar-detect): High-performance star extraction engine. Note that commercial use requires separate licensing from Clear Skies Astro.
- [`starfield`](https://crates.io/crates/starfield): High-precision astronomical time and coordinate precession calculations.

---

## Disclaimer

All product names, trademarks, and registered trademarks are property of their respective owners. All company, product, and service names used in this project are for identification purposes only. Use of these names, trademarks, and brands does not imply endorsement.
