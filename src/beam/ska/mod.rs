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

use std::f64::consts::FRAC_PI_6;

use marlu::RADec;

// These values will be set from ObsContext when needed
pub(crate) struct SkaBeamConfig {
    pub(crate) num_stations: usize,
    pub(crate) phase_centre: RADec,
    pub(crate) ref_freq_hz: f64,
    pub(crate) latitude_rad: f64,
}

impl SkaBeamConfig {
    pub(crate) fn from_obs_context(obs_context: &crate::context::ObsContext) -> Self {
        Self {
            num_stations: obs_context.get_total_num_tiles(),
            phase_centre: obs_context.phase_centre,
            ref_freq_hz: obs_context.fine_chan_freqs[0] as f64,
            latitude_rad: obs_context.array_position.latitude_rad,
        }
    }
}
