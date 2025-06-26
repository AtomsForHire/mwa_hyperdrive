use std::{
    f64::consts::{FRAC_PI_2, PI},
    num,
};

use marlu::{AzEl, Jones, RADec, LMN};
use mwa_hyperbeam::fee;
use ndarray::prelude::*;
use rayon::prelude::*;

use super::SkaBeamParams;
use crate::beam::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::beam::{BeamGpu, DevicePointer, GpuFloat};
use log::{debug, error, warn};
use num_complex::*;
use vec1::Vec1;

const SPEED_OF_LIGHT: f64 = 299792458.0;

#[derive(Clone)]
pub(crate) struct SkaArrayFactorBeam {
    pub phase_centre: RADec,
    pub ska_site_latitude_rad: f64,
    pub reference_frequency_hz: f64,
    pub number_of_stations: usize,
    pub feed_angles_rad: Vec<Vec<f64>>,
    pub feed_coordinates: Vec<Array2<f64>>,
    pub ecef_to_local_mats: Vec<Array2<f64>>,
}

impl SkaArrayFactorBeam {
    pub fn new(params: SkaBeamParams) -> Self {
        Self {
            phase_centre: params.phase_centre,
            ska_site_latitude_rad: params.ska_site_latitude_rad,
            reference_frequency_hz: params.reference_frequency_hz,
            number_of_stations: params.number_of_stations,
            feed_angles_rad: params
                .feed_angles_rad
                .expect("Error! I need feed angles for array factor beam"),
            feed_coordinates: params
                .feed_coordinates
                .expect("Error! I need feed coordinates for array factor beam"),
            ecef_to_local_mats: params.ecef_to_local_mats.expect(
                "Error! I need ecef to local transformation matrices for array factor beam",
            ),
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
        tile_index: Option<usize>,
    ) -> Jones<f64> {
        let index = tile_index.expect("Error! tile_index is needed for array factor beam forming");
        // let index = tile_index.unwrap_or(0 as usize); // Uncomment this for debugging, lets
        // program run all the way through

        // Feed angles, euler angles, azimutal angles from x to y, N of E. Two elements [x, y]
        let phi_pq: &Vec<f64> = &self.feed_angles_rad[index];

        // get element coordinates and transformation matrix for station 'index'
        // NOTE: OSKAR saves element offsets in ECEF coordinates, we need to transform back to
        // local enu coordinates. Fortunately, OSKAR saves the *transpose* of the local to ecef
        // transformation matrix, so we can just multiply the coordinates by the saved matrix.
        let coordinates: &Array2<f64> = &self.feed_coordinates[index];
        let ecef_to_local_mat: &Array2<f64> = &self.ecef_to_local_mats[index];

        // Transform the coordinates
        let transformed_coordinates = coordinates.dot(ecef_to_local_mat);
        let num_elems = coordinates.nrows();

        // Convert frequency to wavelength
        let lambda = SPEED_OF_LIGHT / freq_hz;

        let hadec = azel.to_hadec(self.ska_site_latitude_rad);
        let beam_radec = hadec.to_radec(lst_rad);
        let LMN {
            l: beam_l,
            m: beam_m,
            ..
        } = beam_radec.to_lmn(zenith_radec); // Where the beam is pointing relative to zenith, in
                                             // the (l,m) plane

        // 1. Station rotation
        // The station rotation information, when using the array factor method, is already
        // implicitly included in the coordinates of the elements. We do not need to apply extra
        // rotation for it.
        // NOTE: The array factor is a *scalar* complex quantity, multiply this array factor by the
        // element factor
        // Notation is a bit confusing:
        // 1. We form the array factor with (l, m) coordinates not (theta, phi)
        // 2. station_beam_x_theta is the voltage pattern for the array of x-dipoles
        //    It describes the array's whole x-dipole response to a signal coming from (l, m)
        //    NOTE: But how does it know to describe the response to (x, y) or (theta, phi)
        //    components of the electric field?
        let mut array_factor = Complex::from(0.0);

        for i in 0..num_elems {
            let x_loc = transformed_coordinates[[i, 0]];
            let y_loc = transformed_coordinates[[i, 1]];

            // Add up phases
            let tot_phase = (x_loc / lambda * (beam_l) + y_loc / lambda * (beam_m));
            let angle = -2.0 * PI * tot_phase;
            array_factor += Complex::from_polar(1.0, -angle);
        }

        // Normalise complex Array Factor
        let af = array_factor / num_elems as f64;

        // 1.1 Embedded Element Pattern for crossed dipoles
        // This is assuming the dipoles are aligned with the x and y axis. i.e. NO ROTATION!
        let phi = FRAC_PI_2 - azel.az;
        let theta = FRAC_PI_2 - azel.el;

        // Very much related to SKAO memo written by Randal Wayth
        // Feed angle is a standard counter clokcwise rotation
        // This expression requires a clockwise rotation
        // let ct = theta.cos();
        // let clockwise_rot = PI - feed_angle;
        // let j_ef = Jones::from([
        //     af * (phi + clockwise_rot).cos() * ct * ct,
        //     -af * (phi + clockwise_rot).sin() * ct,
        //     af * (phi + clockwise_rot).sin() * ct * ct,
        //     af * (phi + clockwise_rot).cos() * ct,
        // ]);
        // let j_ef = Jones::from([
        //     af * (phi + clockwise_rot).cos() * ct,
        //     -af * (phi + clockwise_rot).sin(),
        //     af * (phi + clockwise_rot).sin() * ct,
        //     af * (phi + clockwise_rot).cos(),
        // ]);
        // let j_ef = Jones::from([af, af, af, af]);

        // The phi angle is different for both p and q dipoles because q is rotated 90 degrees
        // (usually)
        let phi_p = phi;
        let phi_q = phi + PI / 2.0;
        let denom_p = self.calc_half_wavelength_dipole_denom(theta, phi_p);
        let denom_q = self.calc_half_wavelength_dipole_denom(theta, phi_q);

        let dipole_length: f64 = 0.5;
        let kl: f64 = (dipole_length * (PI * freq_hz / SPEED_OF_LIGHT));
        let numer_p = (kl * phi.cos() * theta.sin()).cos() - kl.cos();
        let numer_q = (kl * (phi_q).cos() * theta.sin()).cos() - kl.cos();

        let e_p_theta = (-phi.cos() * theta.cos() * numer_p) / denom_p * array_factor;
        let e_p_phi = (phi.sin() * numer_p) / denom_p * array_factor;
        let e_q_theta = (-(phi_q).cos() * theta.cos() * numer_q) / denom_q * array_factor;
        let e_q_phi = ((phi_q).sin() * numer_q) / denom_q * array_factor;

        let j_effective = Jones::from([e_p_theta, e_p_phi, e_q_theta, e_q_phi]);
        // 2. Parallactic angle
        // let phi = self.ska_site_latitude_rad;
        // let ha = hadec.ha;
        // let dec = hadec.dec;
        // let psi = (phi.cos() * ha.sin())
        //     .atan2((phi.sin() * dec.cos() - phi.cos() * dec.sin() * ha.cos()));
        //
        // let r_psi = Jones::from([
        //     psi.cos(),
        //     0.0,
        //     -psi.sin(),
        //     0.0,
        //     psi.sin(),
        //     0.0,
        //     psi.cos(),
        //     0.0,
        // ]);
        let r_psi = 1.0;

        return j_effective * r_psi;
    }

    /// Calculate the denominator that is common to both E_phi and E_theta components, when using a
    /// half-wavelength dipole (as OSKAR does)
    fn calc_half_wavelength_dipole_denom(&self, theta: f64, phi: f64) -> f64 {
        return 1.0 + phi.cos() * phi.cos() * (theta.cos() * theta.cos() - 1.0);
    }
}

impl Beam for SkaArrayFactorBeam {
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
        tile_index: Option<usize>,
        lst_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        let zenith_radec = RADec::from_radians(lst_rad, self.ska_site_latitude_rad);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = self.phase_centre.to_lmn(zenith_radec);

        Ok(SkaArrayFactorBeam::calc_jones_inner(
            self,
            azel,
            freq_hz,
            lst_rad,
            zenith_radec,
            cent_l,
            cent_m,
            tile_index,
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
        tile_index: Option<usize>,
        lst_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        let zenith_radec = RADec::from_radians(lst_rad, self.ska_site_latitude_rad);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = self.phase_centre.to_lmn(zenith_radec);

        //println!("IM HERE IM HERE IM HERE TILE_IDX: {:?}", tile_index);

        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                *result = SkaArrayFactorBeam::calc_jones_inner(
                    self,
                    azel,
                    freq_hz,
                    lst_rad,
                    zenith_radec,
                    cent_l,
                    cent_m,
                    tile_index,
                );
            });
        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        // All "tiles" have the same response.
        // TODO: Do I need to change this for rotated stations?
        let tile_map = DevicePointer::copy_to_device(&vec![0; self.number_of_stations])?;
        // Each frequency is distinct.
        let freq_map = DevicePointer::copy_to_device(
            &(0..freqs_hz.len())
                .map(|usize| usize as i32)
                .collect::<Vec<_>>(),
        )?;
        let obj = SkaArrayFactorBeamGpu {
            // cpu_object: *self,
            cpu_object: self.clone(),
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
pub(crate) struct SkaArrayFactorBeamGpu {
    cpu_object: SkaArrayFactorBeam,
    freqs_hz: Vec<u32>,
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for SkaArrayFactorBeamGpu {
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
        self.SkaArrayFactorBeam.station_angle_rad.len();
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
        let beam = SkaArrayFactorBeam;

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
