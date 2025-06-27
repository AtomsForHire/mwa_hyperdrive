// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to abstract beam calculations.
//!
//! [`Beam`] is a trait detailing how to perform various beam-related tasks. By
//! making this trait, we can neatly abstract over multiple beam codes,
//! including a simple [`NoBeam`] type (which just returns identity matrices).
//!
//! Note that (where applicable) `norm_to_zenith` is always true; the
//! implication being that a sky-model source's brightness is always assumed to
//! be correct when at zenith.

mod error;
mod fee;
mod ska;
#[cfg(test)]
mod tests;

pub(crate) use error::BeamError;
pub(crate) use fee::FEEBeam;
pub(crate) use ska::{SkaAiryBeam, SkaArrayFactorBeam, SkaBeamParams, SkaGaussianBeam};

use std::{path::Path, str::FromStr};

use itertools::Itertools;
use log::debug;
use marlu::{AzEl, Jones, RADec};
use ndarray::prelude::*;
use std::f64::consts::PI;
use strum::IntoEnumIterator;

// Default variables for create_beam_object ska beams
const DEFAULT_SKA_PHASE_CENTRE: RADec = RADec { ra: 0.0, dec: 0.0 };
const DEFAULT_SKA_REF_FREQ_HZ: f64 = 100e6;
const DEFAULT_SKA_SITE_LATITUDE_RAD: f64 = 0.0;

#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::gpu::{DevicePointer, GpuFloat};

/// Supported beam types.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    strum_macros::Display,
    strum_macros::EnumIter,
    strum_macros::EnumString,
)]
#[allow(clippy::upper_case_acronyms)]
pub enum BeamType {
    /// Fully-embedded element beam.
    #[strum(serialize = "fee")]
    FEE,

    /// a.k.a. [`NoBeam`]. Only returns identity matrices.
    #[strum(serialize = "none")]
    None,

    #[strum(serialize = "ska_gaussian")]
    SkaGaussian,

    #[strum(serialize = "ska_airy")]
    SkaAiry,

    #[strum(serialize = "ska_array_factor")]
    SkaArrayFactor,
}

impl Default for BeamType {
    fn default() -> Self {
        Self::SkaArrayFactor
    }
}

lazy_static::lazy_static! {
    pub(crate) static ref BEAM_TYPES_COMMA_SEPARATED: String = BeamType::iter().map(|s| s.to_string().to_lowercase()).join(", ");
}

/// A trait abstracting beam code functions.
pub trait Beam: Sync + Send {
    /// Get the type of beam.
    fn get_beam_type(&self) -> BeamType;

    /// Get the number of tiles associated with this beam. This is determined by
    /// how many delays have been provided.
    fn get_num_tiles(&self) -> usize;

    /// Get the dipole delays associated with this beam.
    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>>;

    /// Get the ideal dipole delays associated with this beam.
    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]>;

    /// Get the dipole gains used in this beam object. The rows correspond to
    /// tiles and there are 32 columns, one for each dipole. The first 16 values
    /// are for X dipoles, the second 16 are for Y dipoles.
    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>>;

    /// Get the beam file associated with this beam, if there is one.
    fn get_beam_file(&self) -> Option<&Path>;

    /// Calculate the beam-response Jones matrix for an [`AzEl`] direction. The
    /// delays and gains that will used depend on `tile_index`; if not supplied,
    /// ideal dipole delays and gains are used, otherwise `tile_index` accesses
    /// the information provided when this [`Beam`] was created.
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError>;

    /// Calculate the beam-response Jones matrices for multiple [`AzEl`]
    /// directions. The delays and gains that will used depend on `tile_index`;
    /// if not supplied, ideal dipole delays and gains are used, otherwise
    /// `tile_index` accesses the information provided when this [`Beam`] was
    /// created.
    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError>;

    /// Calculate the beam-response Jones matrices for multiple [`AzEl`]
    /// directions, saving the results into the supplied slice. The slice must
    /// have the same length as `azels`. The delays and gains that will used
    /// depend on `tile_index`; if not supplied, ideal dipole delays and gains
    /// are used, otherwise `tile_index` accesses the information provided when
    /// this [`Beam`] was created.
    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError>;

    /// Given a frequency in Hz, find the closest frequency that the beam code
    /// is defined for. An example of when this is important is with the FEE
    /// beam code, which can only give beam responses at specific frequencies.
    /// On the other hand, the analytic beam can be used at any frequency.
    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64;

    /// If this [`Beam`] supports it, empty the coefficient cache.
    fn empty_coeff_cache(&self);

    #[cfg(any(feature = "cuda", feature = "hip"))]
    /// Using the tile information from this [`Beam`] and frequencies to be
    /// used, return a [`BeamGpu`]. This object only needs frequencies to
    /// calculate beam response [`Jones`] matrices.
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError>;
}

/// A trait abstracting beam code functions on a GPU.
#[cfg(any(feature = "cuda", feature = "hip"))]
pub trait BeamGpu {
    /// Calculate the Jones matrices for each `az` and `za` direction and
    /// frequency (these were defined when the [`BeamCUDA`] was created). The
    /// results are ordered tile, frequency, direction, slowest to fastest.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError>;

    /// Get the type of beam used to create this [`BeamGpu`].
    fn get_beam_type(&self) -> BeamType;

    /// Get a pointer to the device tile map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_tile_map(&self) -> *const i32;

    /// Get a pointer to the device freq map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_freq_map(&self) -> *const i32;

    /// Get the number of de-duplicated tiles associated with this [`BeamGpu`].
    fn get_num_unique_tiles(&self) -> i32;

    /// Get the number of de-duplicated frequencies associated with this
    /// [`BeamGpu`].
    fn get_num_unique_freqs(&self) -> i32;
}

/// An enum to track whether MWA dipole delays are provided and/or necessary.
#[derive(Debug, Clone)]
pub enum Delays {
    /// Delays are fully specified.
    Full(Array2<u32>),

    /// Delays are specified for a single tile. We must assume that these
    /// dipoles apply to all tiles.
    Partial(Vec<u32>),
}

impl Delays {
    /// The delays of some tiles could contain 32 (which means that that
    /// particular dipole is "dead"). It is sometimes useful to get the "ideal"
    /// dipole delays; i.e. what the delays for each tile would be if all
    /// dipoles were alive.
    pub(crate) fn get_ideal_delays(&self) -> [u32; 16] {
        let mut ideal_delays = [32; 16];
        match self {
            Delays::Partial(v) => {
                // There may be 32 elements per row - 16 for X dipoles, 16 for
                // Y. We only want 16, take the mod of the column index.
                v.iter().enumerate().for_each(|(i, &elem)| {
                    ideal_delays[i % 16] = elem;
                });
            }
            Delays::Full(a) => {
                // Iterate over all rows until none of the delays are 32.
                for row in a.outer_iter() {
                    row.iter().enumerate().for_each(|(i, &col)| {
                        let ideal_delay = ideal_delays.get_mut(i % 16).unwrap();

                        // The delays should be the same, modulo some being
                        // 32 (i.e. that dipole's component is dead). This
                        // code will pick the smaller delay of the two
                        // (delays are always <=32). If both are 32, there's
                        // nothing else that can be done.
                        *ideal_delay = (*ideal_delay).min(col);
                    });
                    if ideal_delays.iter().all(|&e| e < 32) {
                        break;
                    }
                }
            }
        }
        ideal_delays
    }

    /// Some tiles' delays might contain 32s (i.e. dead dipoles), and we might
    /// want to ignore that. Take the ideal delays and replace all tiles' delays
    /// with them.
    pub(crate) fn set_to_ideal_delays(&mut self) {
        let ideal_delays = self.get_ideal_delays();
        match self {
            // In this case, the delays are the ideal delays.
            Delays::Full(a) => {
                let ideal_delays = ArrayView1::from(&ideal_delays);
                a.outer_iter_mut().for_each(|mut r| r.assign(&ideal_delays));
            }

            // In this case, no meaningful change can be made.
            Delays::Partial { .. } => (),
        }
    }

    /// Parse user-provided dipole delays.
    pub(crate) fn parse(delays: Vec<u32>) -> Result<Delays, BeamError> {
        if delays.len() != 16 || delays.iter().any(|&v| v > 32) {
            return Err(BeamError::BadDelays);
        }
        Ok(Delays::Partial(delays))
    }
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
pub(crate) struct NoBeam {
    pub(crate) num_tiles: usize,
}

impl Beam for NoBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::None
    }

    fn get_num_tiles(&self) -> usize {
        self.num_tiles
    }

    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]> {
        None
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        None
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    fn get_beam_file(&self) -> Option<&Path> {
        None
    }

    fn calc_jones(
        &self,
        _azel: AzEl,
        _freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        Ok(Jones::identity())
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        _freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        Ok(vec![Jones::identity(); azels.len()])
    }

    fn calc_jones_array_inner(
        &self,
        _azels: &[AzEl],
        _freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        results.fill(Jones::identity());
        Ok(())
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_coeff_cache(&self) {}

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        let obj = NoBeamGpu {
            tile_map: DevicePointer::copy_to_device(&vec![0; self.num_tiles])?,
            freq_map: DevicePointer::copy_to_device(&vec![0; freqs_hz.len()])?,
        };
        Ok(Box::new(obj))
    }
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) struct NoBeamGpu {
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for NoBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        _za_rad: &[GpuFloat],
        _latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        #[cfg(feature = "cuda")]
        use cuda_runtime_sys::{
            cudaMemcpy as gpuMemcpy,
            cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };
        #[cfg(feature = "hip")]
        use hip_sys::hiprt::{
            hipMemcpy as gpuMemcpy, hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };

        let identities: Vec<Jones<GpuFloat>> = vec![Jones::identity(); az_rad.len()];
        gpuMemcpy(
            d_jones,
            identities.as_ptr().cast(),
            identities.len() * std::mem::size_of::<Jones<GpuFloat>>(),
            gpuMemcpyHostToDevice,
        );
        Ok(())
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::None
    }

    fn get_tile_map(&self) -> *const i32 {
        self.tile_map.get()
    }

    fn get_freq_map(&self) -> *const i32 {
        self.freq_map.get()
    }

    fn get_num_unique_tiles(&self) -> i32 {
        1
    }

    fn get_num_unique_freqs(&self) -> i32 {
        1
    }
}

pub fn create_beam_object(
    beam_type: Option<&str>,
    num_tiles: usize,
    dipole_delays: Delays,
) -> Result<Box<dyn Beam>, BeamError> {
    let beam_type = match (
        beam_type,
        beam_type.and_then(|b| BeamType::from_str(b).ok()),
    ) {
        (None, _) => BeamType::default(),
        (Some(_), Some(b)) => b,
        (Some(s), None) => return Err(BeamError::Unrecognised(s.to_string())),
    };

    match beam_type {
        BeamType::None => {
            debug!("Setting up a \"NoBeam\" object");
            Ok(Box::new(NoBeam { num_tiles }))
        }

        BeamType::FEE => {
            debug!("Setting up a FEE beam object");

            // Check that the delays are sensible.
            match &dipole_delays {
                Delays::Partial(v) => {
                    if v.len() != 16 || v.iter().any(|&v| v > 32) {
                        return Err(BeamError::BadDelays);
                    }
                }

                Delays::Full(a) => {
                    if a.len_of(Axis(1)) != 16 || a.iter().any(|&v| v > 32) {
                        return Err(BeamError::BadDelays);
                    }
                    if a.len_of(Axis(0)) != num_tiles {
                        return Err(BeamError::InconsistentDelays {
                            num_rows: a.len_of(Axis(0)),
                            num_tiles,
                        });
                    }
                }
            }

            // Set up the FEE beam struct from the `MWA_BEAM_FILE` environment
            // variable.
            Ok(Box::new(FEEBeam::new_from_env(
                num_tiles,
                dipole_delays,
                None,
            )?))
        }

        BeamType::SkaGaussian => {
            debug!("Setting up a SkaGaussianBeam object via create_beam_object using default SKA params");
            let default_ska_params = SkaBeamParams {
                phase_centre: DEFAULT_SKA_PHASE_CENTRE,
                ska_site_latitude_rad: DEFAULT_SKA_SITE_LATITUDE_RAD,
                reference_frequency_hz: DEFAULT_SKA_REF_FREQ_HZ,
                number_of_stations: num_tiles, // Use num_tiles argument for number_of_stations
                feed_angles_rad: None,
                feed_coordinates: None,
                ecef_to_local_mats: None,
            };
            Ok(Box::new(SkaGaussianBeam::new(default_ska_params)))
        }
        BeamType::SkaAiry => {
            debug!(
                "Setting up a SkaAiryBeam object via create_beam_object using default SKA params"
            );
            let default_ska_params = SkaBeamParams {
                phase_centre: DEFAULT_SKA_PHASE_CENTRE,
                ska_site_latitude_rad: DEFAULT_SKA_SITE_LATITUDE_RAD,
                reference_frequency_hz: DEFAULT_SKA_REF_FREQ_HZ,
                number_of_stations: num_tiles, // Use num_tiles argument for number_of_stations
                feed_angles_rad: None,
                feed_coordinates: None,
                ecef_to_local_mats: None,
            };
            Ok(Box::new(SkaAiryBeam::new(default_ska_params)))
        }
        BeamType::SkaArrayFactor => {
            debug!(
                "Setting up a SkaArrayFactor object via create_beam_object using default SKA params"
            );
            // Populate some default values so I can plot the beam response
            let mut feed_angles_rad: Vec<Vec<f64>> = vec![];
            for i in 0..256 {
                feed_angles_rad.push(vec![0.0, PI / 2.0]);
            }

            let feed_coordinates: Vec<Array2<f64>> = vec![get_s8_1()];
            let ecef_to_local_mats: Vec<Array2<f64>> =
                vec![array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],]];

            let default_ska_params = SkaBeamParams {
                phase_centre: DEFAULT_SKA_PHASE_CENTRE,
                ska_site_latitude_rad: DEFAULT_SKA_SITE_LATITUDE_RAD,
                reference_frequency_hz: DEFAULT_SKA_REF_FREQ_HZ,
                number_of_stations: num_tiles, // Use num_tiles argument for number_of_stations
                feed_angles_rad: None,
                feed_coordinates: None,
                ecef_to_local_mats: None,
            };
            Ok(Box::new(SkaArrayFactorBeam::new(default_ska_params)))
        }
    }
}

fn get_s8_1() -> Array2<f64> {
    let coordinates = array![
        [-15.36, -5.266],
        [-15.137, -9.047],
        [-12.157, -14.172],
        [-10.648, -16.102],
        [-10.56, -13.212],
        [-8.248, -15.407],
        [-6.629, -14.702],
        [-8.399, -12.913],
        [-15.501, -11.765],
        [-13.704, -7.985],
        [-12.514, -9.296],
        [-9.239, -8.54],
        [-10.072, -6.707],
        [-11.934, -7.282],
        [-12.239, -4.324],
        [-13.565, -6.196],
        [-2.76, -15.245],
        [-2.279, -16.986],
        [0.54, -14.194],
        [-2.317, -13.558],
        [1.782, -12.72],
        [-0.713, -12.225],
        [-3.388, -11.963],
        [-5.505, -13.32],
        [-6.744, -11.712],
        [-8.008, -10.463],
        [-9.83, -10.377],
        [-11.688, -11.752],
        [-13.927, -10.758],
        [-6.444, -16.867],
        [-4.93, -15.7],
        [-2.9, -18.73],
        [-5.551, -6.829],
        [-7.284, -6.028],
        [-7.213, -8.436],
        [-9.782, -4.872],
        [-6.977, -4.293],
        [-5.014, -4.743],
        [-5.506, -2.949],
        [-4.16, -0.553],
        [-5.456, -9.115],
        [-4.901, -10.805],
        [-1.865, -10.661],
        [-2.503, -8.925],
        [-1.457, -7.464],
        [-1.03, -5.633],
        [-3.833, -7.358],
        [-2.926, -4.517],
        [6.253, -17.984],
        [6.802, -16.314],
        [3.432, -16.012],
        [5.103, -15.032],
        [5.081, -12.199],
        [3.472, -11.608],
        [3.503, -14.189],
        [1.386, -15.799],
        [-5.717, 0.461],
        [-7.226, -1.787],
        [-8.564, -3.034],
        [-10.459, -0.662],
        [-10.229, -2.399],
        [1.082, -17.547],
        [-0.792, -19.062],
        [3.418, -17.721],
        [1.993, -5.82],
        [0.375, -4.624],
        [0.349, -7.538],
        [7.314, -12.395],
        [7.561, -14.527],
        [9.221, -15.078],
        [10.52, -13.906],
        [12.478, -13.611],
        [0.569, -9.588],
        [1.794, -10.867],
        [2.092, -8.015],
        [4.322, -9.594],
        [6.152, -7.648],
        [3.834, -6.587],
        [4.483, -4.886],
        [1.974, -2.763],
        [10.038, -6.189],
        [8.067, -3.15],
        [10.435, -2.825],
        [9.928, -1.043],
        [8.194, -1.366],
        [6.024, -2.255],
        [3.606, -0.9],
        [3.817, -3.087],
        [9.875, -11.524],
        [9.355, -9.796],
        [7.693, -8.775],
        [7.658, -10.579],
        [6.121, -9.616],
        [6.359, -4.03],
        [6.01, -5.853],
        [7.694, -6.609],
        [11.676, -4.152],
        [11.015, -7.704],
        [9.262, -7.814],
        [13.838, -3.061],
        [14.41, -4.763],
        [16.174, -2.648],
        [18.139, -4.257],
        [18.393, -2.485],
        [11.075, -9.585],
        [11.707, -11.496],
        [14.707, -12.698],
        [13.775, -10.205],
        [17.055, -8.694],
        [13.078, -8.488],
        [15.475, -7.302],
        [13.083, -6.64],
        [14.077, 3.618],
        [15.895, 0.691],
        [17.17, 3.029],
        [16.693, 4.775],
        [19.039, 3.6],
        [17.124, 7.765],
        [15.495, 6.314],
        [10.134, 4.827],
        [17.561, 0.147],
        [18.696, 1.797],
        [14.745, -1.486],
        [12.969, -0.192],
        [11.763, -1.584],
        [11.845, 4.576],
        [12.753, 2.326],
        [14.203, 1.187],
        [10.859, 13.368],
        [9.841, 11.36],
        [8.214, 10.555],
        [4.018, 12.99],
        [6.115, 15.065],
        [5.187, 11.681],
        [7.091, 12.0],
        [9.05, 13.083],
        [8.049, 8.839],
        [11.023, 8.273],
        [12.981, 7.053],
        [12.649, 8.79],
        [14.899, 8.065],
        [13.442, 10.857],
        [11.687, 11.333],
        [13.314, 14.245],
        [6.878, -0.115],
        [8.296, 1.325],
        [11.065, 0.446],
        [10.733, 2.556],
        [10.841, 6.513],
        [7.886, 6.938],
        [8.401, 5.265],
        [5.22, 2.977],
        [7.547, 13.965],
        [10.044, 14.957],
        [7.385, 17.607],
        [5.713, 16.876],
        [4.14, 17.834],
        [3.769, 15.915],
        [5.353, 0.843],
        [7.556, 2.921],
        [2.679, 11.795],
        [1.445, 9.478],
        [0.964, 11.221],
        [-2.732, 15.31],
        [-1.247, 12.793],
        [-1.019, 14.628],
        [0.939, 13.05],
        [2.827, 14.492],
        [2.37, 7.207],
        [2.548, 4.777],
        [4.274, 5.182],
        [4.07, 7.672],
        [6.635, 4.875],
        [6.079, 6.859],
        [6.393, 10.344],
        [3.837, 9.439],
        [1.646, -0.546],
        [2.797, 1.364],
        [0.857, 1.19],
        [3.466, 3.293],
        [1.056, 3.738],
        [-1.301, 1.792],
        [-3.457, 1.111],
        [-2.989, -1.846],
        [0.993, 14.923],
        [1.868, 17.07],
        [0.384, 18.15],
        [-0.906, 16.834],
        [-3.396, 19.051],
        [-1.276, -0.007],
        [-0.818, -3.294],
        [0.42, -1.922],
        [-3.89, 11.224],
        [-5.692, 8.724],
        [-7.397, 9.248],
        [-7.982, 13.667],
        [-8.006, 11.851],
        [-5.691, 10.777],
        [-5.604, 12.881],
        [-3.331, 13.626],
        [-2.594, 2.983],
        [-3.432, 4.976],
        [-1.034, 5.832],
        [0.812, 5.886],
        [-2.628, 7.481],
        [-1.143, 8.552],
        [-1.841, 10.157],
        [-3.883, 8.849],
        [-7.196, 2.04],
        [-4.389, 2.798],
        [-5.147, 4.456],
        [-4.926, 6.286],
        [-6.914, 4.343],
        [-6.764, 6.446],
        [-9.788, 4.551],
        [-10.469, 2.831],
        [-4.866, 15.71],
        [-3.306, 17.08],
        [-5.663, 17.803],
        [-6.781, 15.273],
        [-10.58, 15.331],
        [-11.197, 1.052],
        [-9.119, 1.756],
        [-8.349, -0.366],
        [-12.839, 12.074],
        [-14.696, 10.87],
        [-12.102, 10.384],
        [-17.058, 3.261],
        [-15.866, 1.913],
        [-12.593, 2.375],
        [-14.488, 4.229],
        [-12.715, 5.715],
        [-14.465, 8.938],
        [-11.973, 8.405],
        [-10.994, 6.147],
        [-9.461, 7.2],
        [-9.545, 8.911],
        [-10.521, 11.168],
        [-10.012, 12.931],
        [-11.929, 14.132],
        [-13.321, -1.874],
        [-12.229, -0.396],
        [-13.936, 1.189],
        [-15.381, -0.79],
        [-16.824, 0.326],
        [-18.287, 1.416],
        [-18.972, -0.656],
        [-17.641, -2.379],
        [-15.147, 5.905],
        [-13.561, 7.344],
        [-15.678, 7.637],
        [-17.081, 9.961],
        [-18.327, 5.934],
        [-17.172, -4.909],
        [-16.029, -3.12],
        [-14.016, -3.589],
    ];

    return coordinates;
}
