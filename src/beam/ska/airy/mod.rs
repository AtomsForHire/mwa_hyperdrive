// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::{FRAC_PI_2, PI};

use marlu::{AzEl, Jones, RADec, LMN};
use ndarray::prelude::*;
use rayon::prelude::*;
use num_complex::Complex;

use super::SkaBeamConfig;
use crate::beam::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::beam::{BeamGpu, DevicePointer, GpuFloat};

include!("bindings.rs");

/// `scipy.special.jn_zeros(1, 1)[0] / np.pi`
const J_ZERO_THINGY: f64 = 1.2196698912665045;

const FWHM_RAD: f64 = 0.07452555906;
const FWHM_FACTOR: f64 = 2.35482004503;

#[derive(Clone)]
pub(crate) struct SkaAiryBeam {
    config: SkaBeamConfig,
}

impl SkaAiryBeam {
    pub(crate) fn new(config: SkaBeamConfig) -> Self {
        Self { config }
    }

    /// Calculate the beam response for a single direction
    fn calc_jones_inner(&self, azel: AzEl, freq_hz: f64) -> Jones<f64> {
        let freq_ratio = freq_hz / self.config.ref_freq_hz;
        let fwhm = FWHM_RAD / freq_ratio;
        let sigma = fwhm / FWHM_FACTOR;

        let lmn = azel.to_lmn();
        let l = lmn.l;
        let m = lmn.m;

        // Calculate distance from phase centre
        let r = (l * l + m * m).sqrt();
        if r == 0.0 {
            return Jones::identity();
        }

        // Airy pattern: 2*J1(x)/x where x = pi*D*sin(theta)/lambda
        // For small angles, sin(theta) ≈ theta, and theta = r
        let x = std::f64::consts::PI * r / sigma;
        let j1 = bessel::j1(x);
        let jones = Complex::new(2.0 * j1 / x, 0.0);

        Jones::new(jones, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), jones)
    }
}

impl Beam for SkaAiryBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaAiry
    }

    fn get_num_tiles(&self) -> usize {
        self.config.num_stations
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        Ok(self.calc_jones_inner(azel, freq_hz))
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        Ok(azels
            .par_iter()
            .map(|&azel| self.calc_jones_inner(azel, freq_hz))
            .collect())
    }

    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                *result = self.calc_jones_inner(azel, freq_hz);
            });
        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        // All "tiles" have the same response.
        let tile_map = DevicePointer::copy_to_device(&vec![0; self.config.num_stations])?;
        // Each frequency is distinct.
        let freq_map = DevicePointer::copy_to_device(
            &(0..freqs_hz.len())
                .map(|usize| usize as i32)
                .collect::<Vec<_>>(),
        )?;
        let obj = SkaAiryBeamGpu {
            cpu_object: *self,
            freqs_hz: freqs_hz.to_vec(),
            tile_map,
            freq_map,
        };
        Ok(Box::new(obj))
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        None
    }

    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]> {
        None
    }

    fn get_beam_file(&self) -> Option<&std::path::Path> {
        None
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_coeff_cache(&self) {}
}

#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) struct SkaAiryBeamGpu {
    cpu_object: SkaAiryBeam,
    freqs_hz: Vec<u32>,
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for SkaAiryBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        _latitude_rad: f64,
        _d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        // Not implemented for GPU
        Err(BeamError::Unrecognised("GPU not supported for SKA Airy beam".to_string()))
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaAiry
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
        self.freqs_hz.len() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use marlu::AzEl;

    #[test]
    fn test_airy_calc_jones_inner() {
        let freq_hz = 106000000.;
        let beam = SkaAiryBeam {
            config: SkaBeamConfig {
                num_stations: 1,
                latitude_rad: 0.0,
                phase_centre: LMN::default(),
                ref_freq_hz: 106000000.0,
            },
        };

        let azel = AzEl::from_radians(2.00370398, 1.00922628);
        let jones = beam.calc_jones(azel, freq_hz, None, 0.0).unwrap();
        let expected = 0.0143450805168023_f64.sqrt();
        assert_abs_diff_eq!(jones[0].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[0].im, 0.0);
        assert_abs_diff_eq!(jones[1].re, 0.0);
        assert_abs_diff_eq!(jones[1].im, 0.0);
        assert_abs_diff_eq!(jones[2].re, 0.0);
        assert_abs_diff_eq!(jones[2].im, 0.0);
        assert_abs_diff_eq!(jones[3].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[3].im, 0.0);

        let azel = AzEl::from_radians(0.1, 0.1);
        let jones = beam.calc_jones(azel, freq_hz, None, 0.0).unwrap();
        let expected = 1.20727184e-5_f64.sqrt();
        assert_abs_diff_eq!(jones[0].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[0].im, 0.0);
        assert_abs_diff_eq!(jones[1].re, 0.0);
        assert_abs_diff_eq!(jones[1].im, 0.0);
        assert_abs_diff_eq!(jones[2].re, 0.0);
        assert_abs_diff_eq!(jones[2].im, 0.0);
        assert_abs_diff_eq!(jones[3].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[3].im, 0.0);
    }
}
