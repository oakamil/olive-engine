use ndarray::Array2;
use olive_engine::CameraTrait;
use olive_engine::uvc_camera::{Camera, UvcControls};
use std::time::{Duration, Instant};
use tetra3::{
    fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor},
    solver::{SolveOptions, SolveStatus},
    tetra3::Tetra3,
};

#[derive(Clone, Debug)]
struct TestResult {
    combo: UvcControls,
    solves: usize,
    total: usize,
    solve_rate: f64,
    avg_centroids: f64,
}

const TUNE_SATURATION: bool = false;
const TUNE_CONTRAST: bool = true;
const TUNE_WHITE_BALANCE: bool = false;
const TUNE_SHARPNESS: bool = false;
const TUNE_BRIGHTNESS: bool = false;
const TUNE_GAMMA: bool = true;

fn get_choices(range: &Option<uvc::Range<i32>>) -> Vec<Option<i32>> {
    match range {
        Some(r) => {
            let mut choices = vec![r.min, r.max];
            if r.default >= r.min && r.default <= r.max {
                choices.push(r.default);
            }
            choices.sort();
            choices.dedup();
            choices.into_iter().map(Some).collect()
        }
        None => vec![None],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_fps: u32 = if args.len() > 1 {
        args[1].parse().unwrap_or(10)
    } else {
        10
    };
    let user_gain: Option<f32> = if args.len() > 2 {
        args[2].parse().ok()
    } else {
        None
    };

    println!("Initializing UVC Tuner with target {} fps...", target_fps);
    let mut camera = Camera::new(target_fps, false);

    let mut start_attempts = 0;
    while let Err(e) = camera.start_capture() {
        start_attempts += 1;
        if start_attempts >= 3 {
            eprintln!("Failed to start capture after 3 attempts: {}", e);
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    // Wait for the camera thread to populate the ranges, up to 3 attempts
    let mut wait_attempts = 0;
    while camera.get_config().is_none() && wait_attempts < 4 {
        std::thread::sleep(Duration::from_secs(1));
        wait_attempts += 1;
    }

    if let Some(config) = camera.get_config() {
        println!(
            "Camera configured: {}x{} at {} fps",
            config.width, config.height, config.fps
        );
    } else {
        println!("Warning: Could not fetch camera configuration.");
    }

    println!("Waiting 3s for stream to stabilize before running benchmarks...");
    std::thread::sleep(Duration::from_secs(3));

    let ranges = {
        let lock = camera.ranges.lock().unwrap();
        if let Some(r) = &*lock {
            r.clone()
        } else {
            eprintln!("Failed to retrieve UVC ranges from the camera.");
            return;
        }
    };

    println!("Detected UVC Ranges: {:?}", ranges);

    let sat_choices = if TUNE_SATURATION {
        get_choices(&ranges.saturation)
    } else {
        vec![ranges.saturation.as_ref().map(|r| r.default)]
    };
    let cont_choices = if TUNE_CONTRAST {
        get_choices(&ranges.contrast)
    } else {
        vec![ranges.contrast.as_ref().map(|r| r.default)]
    };
    let wbt_choices = if TUNE_WHITE_BALANCE {
        get_choices(&ranges.white_balance_temperature)
    } else {
        vec![ranges.white_balance_temperature.as_ref().map(|r| r.default)]
    };
    let sharp_choices = if TUNE_SHARPNESS {
        get_choices(&ranges.sharpness)
    } else {
        vec![ranges.sharpness.as_ref().map(|r| r.default)]
    };
    let bright_choices = if TUNE_BRIGHTNESS {
        get_choices(&ranges.brightness)
    } else {
        vec![ranges.brightness.as_ref().map(|r| r.default)]
    };
    let gamma_choices = if TUNE_GAMMA {
        get_choices(&ranges.gamma)
    } else {
        vec![ranges.gamma.as_ref().map(|r| r.default)]
    };

    let mut combinations = Vec::new();
    for &sat in &sat_choices {
        for &cont in &cont_choices {
            for &wbt in &wbt_choices {
                for &sharp in &sharp_choices {
                    for &bright in &bright_choices {
                        for &gamma in &gamma_choices {
                            combinations.push(UvcControls {
                                ae_mode: None,
                                exposure_abs: None,
                                saturation: sat,
                                contrast: cont,
                                white_balance_temperature: wbt,
                                sharpness: sharp,
                                brightness: bright,
                                gamma,
                                gain: None,
                            });
                        }
                    }
                }
            }
        }
    }

    println!("Testing {} combinations...", combinations.len());

    // Load Tetra3 solver
    let db_path = format!(
        "{}/data/default_database.npz",
        std::env::var("HOME").unwrap_or_default()
    );
    let mut solver = Tetra3::new(&db_path);

    let mut results = Vec::new();
    let mut extractor: Option<FastExtractor> = None;

    // Set Gain for the test
    let test_gain = user_gain.unwrap_or_else(|| camera.get_max_gain());
    println!("Setting camera gain for tests: {}", test_gain);
    camera.set_gain(test_gain);

    for (i, combo) in combinations.iter().enumerate() {
        println!("Test {}/{}: {:?}", i + 1, combinations.len(), combo);

        // Apply settings
        camera.set_uvc_controls(combo.clone());

        // Wait for settings to apply before reading frames
        std::thread::sleep(Duration::from_secs(1));

        let start_time = Instant::now();
        let mut solves = 0;
        let mut total = 0;
        let mut total_centroids = 0;
        let mut last_timestamp = None;

        while start_time.elapsed() < Duration::from_secs(3) {
            let img_arc = camera.wait_for_next_capture(last_timestamp);
            last_timestamp = Some(img_arc.timestamp);
            let img_data = &img_arc.data;

            let config = camera.get_config().expect("Camera config missing");
            let width = config.width as usize;
            let height = config.height as usize;

            let array_img = match Array2::from_shape_vec((height, width), img_data.clone()) {
                Ok(arr) => arr,
                Err(_) => continue,
            };

            if extractor.is_none() {
                let downsample = if width * height > 1_000_000 {
                    FastDownsample::X2
                } else {
                    FastDownsample::None
                };

                let options = FastExtractOptions {
                    sigma: 3.5,
                    downsample,
                    bg_sub_mode: Some(FastBgSubMode::LineMedian),
                    binary_open: false,
                    min_area: Some(1),
                    max_area: Some(20),
                    ..FastExtractOptions::default()
                };
                extractor = Some(FastExtractor::new(width, height, options));
            }

            let ext = extractor.as_mut().unwrap();
            let result = ext.extract_sequential(&array_img);

            let num_centroids = result.len();
            total_centroids += num_centroids;

            let mut centroids_arr = Array2::<f64>::zeros((num_centroids, 2));
            for (i, c) in result.iter().enumerate() {
                centroids_arr[[i, 0]] = c.y;
                centroids_arr[[i, 1]] = c.x;
            }

            let size = (height as f64, width as f64);
            if num_centroids >= 4 {
                if let Ok(solution) =
                    solver.solve_from_centroids(&centroids_arr, size, SolveOptions::default())
                {
                    if matches!(solution.status, SolveStatus::MatchFound) {
                        solves += 1;
                    }
                }
            }
            total += 1;
        }

        let solve_rate = if total > 0 {
            (solves as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let avg_centroids = if total > 0 {
            total_centroids as f64 / total as f64
        } else {
            0.0
        };

        println!(
            "Result: {} solves out of {} frames ({:.2}%), avg centroids: {:.1}",
            solves, total, solve_rate, avg_centroids
        );

        results.push(TestResult {
            combo: combo.clone(),
            solves,
            total,
            solve_rate,
            avg_centroids,
        });
    }

    // Rank results by solve rate descending, then by avg centroids descending
    results.sort_by(|a, b| {
        b.solve_rate
            .partial_cmp(&a.solve_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.avg_centroids
                    .partial_cmp(&a.avg_centroids)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    println!("\n=== TOP 20 COMBINATIONS ===");
    for (i, res) in results.iter().take(20).enumerate() {
        println!(
            "{}. Solve Rate: {:.2}% ({}/{}) | Avg Centroids: {:.1} | Combo: {:?}",
            i + 1,
            res.solve_rate,
            res.solves,
            res.total,
            res.avg_centroids,
            res.combo
        );
    }
}
