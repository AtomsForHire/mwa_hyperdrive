// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::{FRAC_PI_2, PI};

use marlu::{AzEl, Jones, RADec, LMN};
use mwa_hyperbeam::fee;
use ndarray::prelude::*;
use rayon::prelude::*;

use super::SkaBeamParams;
// use super::{NUM_STATIONS, PHASE_CENTRE, REF_FREQ_HZ, SKA_LATITUDE_RAD};
use crate::beam::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::beam::{BeamGpu, DevicePointer, GpuFloat};
use env_logger::warn;

include!("bindings.rs");

/// `scipy.special.jn_zeros(1, 1)[0] / np.pi`
const J_ZERO_THINGY: f64 = 1.2196698912665045;

// lazy_static::lazy_static! {
//     static ref AIRY_CONST: f64 = PI * J_ZERO_THINGY / (5.15_f64.to_radians() * REF_FREQ_HZ);
// }

#[derive(Clone, Copy)]
pub(crate) struct SkaAiryBeam {
    pub phase_centre: RADec,
    pub ska_site_latitude_rad: f64,
    pub reference_frequency_hz: f64,
    pub number_of_stations: usize,
    pub station_angle_rad: Vec1<f64>,
    pub feed_angle_rad: Vec1<f64>,
}

impl SkaAiryBeam {
    pub fn new(params: SkaBeamParams) -> Self {
        Self {
            phase_centre: params.phase_centre,
            ska_site_latitude_rad: params.ska_site_latitude_rad,
            reference_frequency_hz: params.reference_frequency_hz,
            number_of_stations: params.number_of_stations,
            station_angle_rad: params.station_angle_rad,
            feed_angle_rad: params.feed_angle_rad,
        }
    }

    fn calc_jones_inner(
        &self,
        azel: AzEl,
        freq_hz: f64,
        lst_rad: f64,
        zenith_radec: RADec,
        cent_l: f64,
        cent_m: f64,
        tile_index: Opntion<usize>,
    ) -> Jones<f64> {
        let index = if let Some(i) = tile_index {
            i
        } else {
            warn!("Warning tile index is needed for Airy beam forming!");
        };

        let station_angle = self.station_angle_rad[i];
        let feed_angle = self.feed_angle_rad[i];

        let airy_const: f64 =
            PI * J_ZERO_THINGY / (5.15_f64.to_radians() * self.reference_frequency_hz);
        let hadec = azel.to_hadec(self.ska_site_latitude_rad);
        let beam_radec = hadec.to_radec(lst_rad);
        let LMN {
            l: beam_l,
            m: beam_m,
            ..
        } = beam_radec.to_lmn(zenith_radec);

        // Original l, m relative to phase centre
        let l_prime = beam_l - cent_l;
        let m_prime = beam_m - cent_m;

        // Rotate source position into rotated station's coordinate frame
        let l_station_frame = l_prime * station_angle.cos() + m_prime * station_angle.sin();
        let m_station_frame = -l_prime * station_angle.sin() + m_prime * station_angle.cos();

        // let dist = ((beam_l - cent_l).powi(2) + (beam_m - cent_m).powi(2)).sqrt();
        let dist = (l_station_frame.powi(2) + m_station_frame.powi(2)).sqrt();

        // More explicit.
        // let radius = 5.15_f64.to_radians() * REF_FREQ_HZ / freq_hz;
        // let rt = dist / (radius / J_ZERO_THINGY) * PI;
        let rt = dist * freq_hz * airy_const;

        // This takes into account the *station* rotation
        let z = (2.0 * unsafe { j1(rt) } / rt).abs();

        // Need to calculate parallactic angle from feed angle
        // TODO: This part done with LLM, need to double check
        let ha_rad = hadec.ha;
        let dec_rad = hadec.dec;
        let lat_rad = self.ska_site_latitude_rad;

        let sin_h = ha_rad.sin();
        let cos_h = ha_rad.cos();
        let sin_d = dec_rad.sin();
        let cos_d = dec_rad.cos();
        let tan_l = lat_rad.tan();

        // Parallactic angle psi
        let parallactic_angle_rad = sin_h.atan2(tan_l * cos_d - sin_d * cos_h);

        // Angle used for rotation
        // let effective_angle = parallactic_angle_rad - feed_angle;

        // Create rotation matrix from Jones type, since multiplication is defined already
        let r_feed = Jones::from([
            feed_angle.cos(),
            -feed_angle.sin(),
            feed_angle.sin(),
            feed_angle.cos(),
        ]);

        let r_parallactic = Jones::from([
            parallactic_angle_rad.cos(),
            -parallactic_angle_rad.sin(),
            parallactic_angle_rad.sin(),
            parallactic_angle_rad.cos(),
        ]);

        // This is the initial Jones matrix. How the X and Y dipoles are
        let j_initial = Jones::from([z, 0.0, 0.0, 0.0, 0.0, 0.0, z, 0.0]);

        r_feed * r_parallactic * j_initial
    }
}

impl Beam for SkaAiryBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaAiry
    }

    fn get_num_tiles(&self) -> usize {
        self.number_of_stations
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    /// Derived with help from Jack Line, Dev Null and
    /// <https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#AiryDisk2D.evaluate>
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        _tile_index: Option<usize>,
        lst_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        let zenith_radec = RADec::from_radians(lst_rad, self.ska_site_latitude_rad);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = self.phase_centre.to_lmn(zenith_radec);

        Ok(SkaAiryBeam::calc_jones_inner(
            self,
            azel,
            freq_hz,
            lst_rad,
            zenith_radec,
            cent_l,
            cent_m,
        ))
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        let mut results = vec![Jones::default(); azels.len()];
        self.calc_jones_array_inner(azels, freq_hz, tile_index, latitude_rad, &mut results)?;
        Ok(results)
    }

    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        _tile_index: Option<usize>,
        lst_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        let zenith_radec = RADec::from_radians(lst_rad, self.ska_site_latitude_rad);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = self.phase_centre.to_lmn(zenith_radec);

        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                *result = SkaAiryBeam::calc_jones_inner(
                    self,
                    azel,
                    freq_hz,
                    lst_rad,
                    zenith_radec,
                    cent_l,
                    cent_m,
                );
            });
        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        // All "tiles" have the same response.
        let tile_map = DevicePointer::copy_to_device(&vec![0; self.number_of_stations])?;
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
        latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        let lst_rad = latitude_rad;

        #[cfg(all(any(feature = "cuda", feature = "hip"), not(feature = "gpu-single")))]
        let azels = az_rad
            .iter()
            .zip(za_rad.iter())
            .map(|(&az, &za)| AzEl::from_radians(az, FRAC_PI_2 - za))
            .collect::<Vec<_>>();
        #[cfg(feature = "gpu-single")]
        let azels = az_rad
            .iter()
            .zip(za_rad.iter())
            .map(|(&az, &za)| AzEl::from_radians(az as f64, FRAC_PI_2 - za as f64))
            .collect::<Vec<_>>();

        let mut a: Array2<Jones<GpuFloat>> = Array2::zeros((self.freqs_hz.len(), az_rad.len()));
        #[cfg(feature = "gpu-single")]
        let mut v = vec![Jones::default(); az_rad.len()];
        for (mut a, &freq) in a.outer_iter_mut().zip(self.freqs_hz.iter()) {
            let freq = f64::from(freq);

            cfg_if::cfg_if! {
                if #[cfg(feature = "gpu-single")] {
                    self.cpu_object
                        .calc_jones_array_inner(&azels, freq, None, lst_rad, &mut v)?;
                    a.iter_mut()
                        .zip(v.iter())
                        .for_each(|(a, v)| *a = Jones::<f32>::from(*v));
                } else {
                    let a = a
                        .as_slice_mut()
                        .expect("cannot fail as memory is contiguous");
                    self.cpu_object
                        .calc_jones_array_inner(&azels, freq, None, lst_rad, a)?;
                }
            }
        }

        #[cfg(feature = "cuda")]
        use cuda_runtime_sys::{
            cudaMemcpy as gpuMemcpy,
            cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };
        #[cfg(feature = "hip")]
        use hip_sys::hiprt::{
            hipMemcpy as gpuMemcpy, hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };
        gpuMemcpy(
            d_jones,
            a.as_ptr().cast(),
            a.len() * std::mem::size_of::<Jones<GpuFloat>>(),
            gpuMemcpyHostToDevice,
        );
        crate::gpu::check_for_errors(crate::gpu::GpuCall::CopyToDevice)?;

        Ok(())
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
        let lst_rad = 5.769848203643869;
        let beam = SkaAiryBeam;

        let azel = AzEl::from_radians(2.00370398, 1.00922628);
        let jones = beam.calc_jones(azel, freq_hz, None, lst_rad).unwrap();
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
        let jones = beam.calc_jones(azel, freq_hz, None, lst_rad).unwrap();
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
