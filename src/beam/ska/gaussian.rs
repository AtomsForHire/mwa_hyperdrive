// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::FRAC_PI_2;

use marlu::{AzEl, Jones, RADec, LMN};
use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

use super::SkaBeamConfig;
use crate::beam::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::beam::{BeamGpu, DevicePointer, GpuFloat};

const FWHM_RAD: f64 = 0.07452555906;
const FWHM_FACTOR: f64 = 2.35482004503;

#[derive(Clone)]
pub(crate) struct SkaGaussianBeam {
    config: SkaBeamConfig,
}

impl SkaGaussianBeam {
    pub(crate) fn new(config: SkaBeamConfig) -> Self {
        Self { config }
    }

    /// Explicitly a 2D gaussian function
    #[allow(clippy::too_many_arguments)]
    fn gaussian_2d(
        x: f64,
        y: f64,
        x0: f64,
        y0: f64,
        sigma_x: f64,
        sigma_y: f64,
        amplitude: f64,
    ) -> f64 {
        let x_diff = x - x0;
        let y_diff = y - y0;
        amplitude
            * (-0.5
                * (x_diff * x_diff / (sigma_x * sigma_x) + y_diff * y_diff / (sigma_y * sigma_y)))
            .exp()
    }

    /// Calculate the beam response for a single direction
    fn calc_jones_inner(&self, azel: AzEl, freq_hz: f64) -> Jones<f64> {
        let freq_ratio = freq_hz / self.config.ref_freq_hz;
        let fwhm = FWHM_RAD / freq_ratio;
        let sigma = fwhm / FWHM_FACTOR;

        let lmn = azel.to_lmn();
        let l = lmn.l;
        let m = lmn.m;

        let jones = Complex::new(
            self.gaussian_2d(l, m, 0.0, 0.0, sigma, sigma, 1.0),
            0.0,
        );

        Jones::new(jones, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), jones)
    }
}

impl Beam for SkaGaussianBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaGaussian
    }

    fn get_num_tiles(&self) -> usize {
        self.config.num_stations
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

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
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
impl BeamGpu for SkaGaussianBeam {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        _latitude_rad: f64,
        _d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        // Not implemented for GPU
        Err(BeamError::Unrecognised("GPU not supported for SKA Gaussian beam".to_string()))
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaGaussian
    }

    fn get_tile_map(&self) -> *const i32 {
        std::ptr::null()
    }

    fn get_freq_map(&self) -> *const i32 {
        std::ptr::null()
    }

    fn get_num_unique_tiles(&self) -> i32 {
        0
    }

    fn get_num_unique_freqs(&self) -> i32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use marlu::AzEl;

    #[test]
    fn test_gaussian_2d() {
        let std = 0.031618803234858744;
        let cent_l = 0.4252937845833011;
        let cent_m = -0.10576131883022044;
        let beam_l = 0.48339108;
        let beam_m = -0.22339675;
        let beam_real = SkaGaussianBeam::gaussian_2d(beam_l, beam_m, cent_l, cent_m, std, std, 1.0);
        assert_abs_diff_eq!(beam_real, 0.00018248210368566883, epsilon = 1e-6);
    }

    #[test]
    fn test_gaussian_calc_jones_inner() {
        let freq_hz = 106000000.;
        let lst_rad = 5.769848203643869;
        let beam = SkaGaussianBeam {
            config: SkaBeamConfig {
                num_stations: 1,
                ref_freq_hz: freq_hz,
            },
        };

        let azel = AzEl::from_radians(2.00370398, 1.00922628);
        let jones = beam.calc_jones(azel, freq_hz, None, lst_rad).unwrap();
        let expected = 0.00018248210368566883;
        assert_abs_diff_eq!(jones[0], Complex::new(expected, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[1], Complex::new(0.0, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[2], Complex::new(0.0, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[3], Complex::new(expected, 0.0), epsilon = 1e-6);
    }
}
