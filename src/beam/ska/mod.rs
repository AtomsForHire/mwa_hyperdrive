// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for SKA-Low beam calculations. A significant amount of this code was
//! written by Dev Null. This code is intended to be a dirty-one-off for SDC3.
//!
//! Warning: `latitude_rad` is used for LST.

mod airy;
mod array_factor;
mod gaussian;

pub(crate) use airy::SkaAiryBeam;
pub(crate) use array_factor::SkaArrayFactorBeam;
pub(crate) use gaussian::SkaGaussianBeam;

use std::f64::consts::FRAC_PI_6;

use marlu::RADec;

// const NUM_STATIONS: usize = 512;
// const PHASE_CENTRE: RADec = RADec {
//     ra: 0.0,
//     dec: -FRAC_PI_6,
// };
// const REF_FREQ_HZ: f64 = 106e6;
// const SKA_LATITUDE_RAD: f64 = -0.4681797212;

#[derive(Clone, Debug)]
pub struct SkaBeamParams {
    pub phase_centre: RADec,
    pub ska_site_latitude_rad: f64,
    pub reference_frequency_hz: f64,
    pub number_of_stations: usize,
    pub feed_angles_rad: Option<Vec<Vec<f64>>>,
    pub feed_coordinates: Option<Vec<ndarray::Array2<f64>>>,
    pub ecef_to_local_mats: Option<Vec<ndarray::Array2<f64>>>,
}
