// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for SKA-Low beam calculations. A significant amount of this code was
//! written by Dev Null. This code is intended to be a dirty-one-off for SDC3.
//!
//! Warning: `latitude_rad` is used for LST.

mod airy;
mod gaussian;

pub(crate) use airy::SkaAiryBeam;
pub(crate) use gaussian::SkaGaussianBeam;

use std::f64::consts::FRAC_PI_2;

use marlu::{AzEl, Jones, RADec};
use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

use crate::beam::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::beam::{BeamGpu, DevicePointer, GpuFloat};

/// Configuration for SKA beams
#[derive(Debug, Clone)]
pub struct SkaBeamConfig {
    /// Number of stations in the array
    pub num_stations: usize,
    /// Phase centre of the observation
    pub phase_centre: RADec,
    /// Reference frequency in Hz
    pub ref_freq_hz: f64,
    /// Array latitude in radians
    pub latitude_rad: f64,
}

impl SkaBeamConfig {
    /// Create a new SKA beam configuration
    pub fn new(num_stations: usize, phase_centre: RADec, ref_freq_hz: f64, latitude_rad: f64) -> Self {
        Self {
            num_stations,
            phase_centre,
            ref_freq_hz,
            latitude_rad,
        }
    }
}
