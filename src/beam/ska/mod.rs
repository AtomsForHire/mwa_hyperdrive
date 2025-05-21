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
use marlu::RADec;

// use std::f64::consts::FRAC_PI_6;
//
// const NUM_STATIONS: usize = 512;
// const PHASE_CENTRE: RADec = RADec {
//     ra: 0.0,
//     dec: -FRAC_PI_6,
// };
// const REF_FREQ_HZ: f64 = 106e6;
// const SKA_LATITUDE_RAD: f64 = -0.4681797212;

pub struct SkaBeamParams {
    pub num_stations: usize,
    pub phase_centre: RADec,
    pub ref_freq_hz: f64,
    pub ska_latitude_rad: f64,
}

/// A trait abstracting beam code functions.
pub trait SkaBeam: Sync + Send {
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
        beam_params: SkaBeamParams,
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
        beam_params: SkaBeamParams,
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
        beam_params: SkaBeamParams,
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
