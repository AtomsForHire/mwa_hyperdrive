// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    path::PathBuf,
    thread::{self, ScopedJoinHandle},
};

use crossbeam_channel::{unbounded, Sender};
use crossbeam_utils::atomic::AtomicCell;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::izip;
use log::{debug, info, log_enabled, Level::Debug};
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR},
    Jones,
};
use ndarray::{iter::AxisIterMut, prelude::*, ArcArray2};
use rayon::prelude::*;
use scopeguard::defer_on_unwind;
use vec1::Vec1;

use super::{InputVisParams, ModellingParams, OutputVisParams};
use crate::{
    averaging::Timeblock,
    beam::Beam,
    context::Polarisations,
    di_calibrate::{calibrate_timeblocks, IncompleteSolutions},
    io::{
        read::VisReadError,
        write::{write_vis, VisTimestep, VisWriteError},
    },
    misc::expensive_op,
    model::{new_sky_modeller, ModelError, SkyModeller},
    solutions::CalSolutionType,
    srclist::SourceList,
    CalibrationSolutions, PROGRESS_BARS,
};

/// Parameters needed to perform calibration.
pub(crate) struct DiCalParams {
    /// The interface to the input data, metadata and flags.
    pub(crate) input_vis_params: InputVisParams,

    /// Beam object.
    pub(crate) beam: Box<dyn Beam>,

    /// The sky-model source list.
    pub(crate) source_list: SourceList,

    /// Blocks of timesteps used for calibration. Each timeblock contains
    /// indices of the input data to average together during calibration. Each
    /// timeblock may have a different number of timesteps; the number of blocks
    /// and their lengths depends on which input data timesteps are being used
    /// as well as the `time_average_factor` (i.e. the number of timesteps to
    /// average during calibration; by default we average all timesteps).
    ///
    /// Simple examples: If we are averaging all data over time to form
    /// calibration solutions, there will only be one timeblock, and that block
    /// will contain all input data timestep indices. On the other hand, if
    /// `time_average_factor` is 1, then there are as many timeblocks as there
    /// are timesteps, and each block contains 1 timestep index.
    ///
    /// A more complicated example: If we are using input data timesteps 10, 11,
    /// 12 and 15 with a `time_average_factor` of 4, then there will be 2
    /// timeblocks, even though there are only 4 timesteps. This is because
    /// timestep 10 and 15 can't occupy the same timeblock with the "length" is
    /// 4. So the first timeblock contains 10, 11 and 12, whereas the second
    /// contains only 15.
    pub(crate) cal_timeblocks: Vec1<Timeblock>,

    /// The minimum UVW cutoff used in calibration \[metres\].
    pub(crate) uvw_min: f64,

    /// The maximum UVW cutoff used in calibration \[metres\].
    pub(crate) uvw_max: f64,

    /// The centroid frequency of the observation used to convert UVW cutoffs
    /// specified in lambdas to metres \[Hz\].
    pub(crate) freq_centroid: f64,

    /// Multiplicative factors to apply to unflagged baselines. These are mostly
    /// all 1.0, but flagged baselines (perhaps due to a UVW cutoff) have values
    /// of 0.0.
    pub(crate) baseline_weights: Vec1<f64>,

    /// The maximum number of times to iterate when performing calibration.
    pub(crate) max_iterations: u32,

    /// The threshold at which we stop convergence when performing calibration.
    /// This is smaller than `min_threshold`.
    pub(crate) stop_threshold: f64,

    /// The minimum threshold to satisfy convergence when performing calibration.
    /// Reaching this threshold counts as "converged", but it's not as good as
    /// the stop threshold. This is bigger than `stop_threshold`.
    pub(crate) min_threshold: f64,

    /// The paths to the files where the calibration solutions are written. The
    /// same solutions are written to each file here, but the format may be
    /// different (indicated by the second part of the tuples).
    pub(crate) output_solution_files: Vec1<(PathBuf, CalSolutionType)>,

    /// The parameters for optional sky-model visibilities files. If specified,
    /// model visibilities will be written out before calibration.
    pub(crate) output_model_vis_params: Option<OutputVisParams>,

    /// Parameters for modelling.
    pub(crate) modelling_params: ModellingParams,
}

impl DiCalParams {
    /// Use the [`DiCalParams`] to perform calibration and obtain solutions.
    pub(crate) fn run(&self) -> Result<CalibrationSolutions, DiCalibrateError> {
        let input_vis_params = &self.input_vis_params;
        let num_cal_timeblocks = self.cal_timeblocks.len();
        let num_chanblocks = input_vis_params.spw.chanblocks.len();
        let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();

        if log_enabled!(Debug) {
            let shape = (num_cal_timeblocks, num_unflagged_tiles, num_chanblocks);
            debug!(
                "Shape of DI Jones matrices array: ({} timeblocks, {} tiles, {} chanblocks; {} MiB)",
                shape.0,
                shape.1,
                shape.2,
                shape.0 * shape.1 * shape.2 * std::mem::size_of::<Jones<f64>>()
                // 1024 * 1024 == 1 MiB.
                / 1024 / 1024
            );
        }

        // Stream one calibration timeblock at a time when that reduces peak host
        // RAM (multiple cal timeblocks) and we are not also writing model vis
        // (writer expects a single continuous model stream).
        let stream_cal_timeblocks =
            self.cal_timeblocks.len() > 1 && self.output_model_vis_params.is_none();

        let (sols, results) = if stream_cal_timeblocks {
            info!(
                "Streaming {} calibration timeblocks to limit peak host memory",
                self.cal_timeblocks.len()
            );
            let shape = (num_cal_timeblocks, num_unflagged_tiles, num_chanblocks);
            let mut di_jones = Array3::from_elem(shape, Jones::identity());
            let mut all_cal_results = Vec::with_capacity(num_cal_timeblocks * num_chanblocks);

            for cal_tb in &self.cal_timeblocks {
                info!(
                    "Loading data and model for calibration timeblock {}/{}",
                    cal_tb.index + 1,
                    num_cal_timeblocks
                );
                let cal_vis = self.get_cal_vis_for_input_range(cal_tb.range.clone())?;
                assert_eq!(cal_vis.vis_weights.len_of(Axis(2)), self.baseline_weights.len());

                let local_tb = Timeblock {
                    index: 0,
                    range: 0..cal_vis.vis_data.len_of(Axis(0)),
                    timestamps: cal_tb.timestamps.clone(),
                    timesteps: cal_tb.timesteps.clone(),
                    median: cal_tb.median,
                };
                let local_tbs = Vec1::new(local_tb);
                let (incomplete, results) = calibrate_timeblocks(
                    cal_vis.vis_data.view(),
                    cal_vis.vis_model.view(),
                    &local_tbs,
                    &input_vis_params.spw.chanblocks,
                    self.max_iterations,
                    self.stop_threshold,
                    self.min_threshold,
                    cal_vis.pols,
                    true,
                );
                di_jones
                    .slice_mut(s![cal_tb.index, .., ..])
                    .assign(&incomplete.di_jones.slice(s![0, .., ..]));
                all_cal_results.append(&mut results.into_raw_vec_and_offset().0);
            }

            let cal_results = Array2::from_shape_vec(
                (num_cal_timeblocks, num_chanblocks),
                all_cal_results,
            )
            .expect("cal results length matches timeblocks × chanblocks");

            let incomplete = IncompleteSolutions {
                di_jones,
                timeblocks: &self.cal_timeblocks,
                chanblocks: &input_vis_params.spw.chanblocks,
                max_iterations: self.max_iterations,
                stop_threshold: self.stop_threshold,
                min_threshold: self.min_threshold,
            };
            (incomplete, cal_results)
        } else {
            let CalVis {
                vis_data,
                vis_weights,
                vis_model,
                pols,
            } = self.get_cal_vis()?;
            assert_eq!(vis_weights.len_of(Axis(2)), self.baseline_weights.len());

            calibrate_timeblocks(
                vis_data.view(),
                vis_model.view(),
                &self.cal_timeblocks,
                &input_vis_params.spw.chanblocks,
                self.max_iterations,
                self.stop_threshold,
                self.min_threshold,
                pols,
                true,
            )
        };

        // "Complete" the solutions.
        let sols = sols.into_cal_sols(self, Some(results.map(|r| r.max_precision)));

        Ok(sols)
    }

    /// For calibration, read in unflagged visibilities and generate sky-model
    /// visibilities for all selected input timeblocks.
    pub(crate) fn get_cal_vis(&self) -> Result<CalVis, DiCalibrateError> {
        self.get_cal_vis_for_input_range(0..self.input_vis_params.timeblocks.len())
    }

    /// Like [`Self::get_cal_vis`], but only for a contiguous range of indices
    /// into [`InputVisParams::timeblocks`]. Used to stream calibration
    /// timeblocks and limit peak host memory.
    pub(crate) fn get_cal_vis_for_input_range(
        &self,
        input_time_range: std::ops::Range<usize>,
    ) -> Result<CalVis, DiCalibrateError> {
        let input_vis_params = &self.input_vis_params;
        let selected_timeblocks = &input_vis_params.timeblocks[input_time_range.clone()];
        if selected_timeblocks.is_empty() {
            return Err(DiCalibrateError::InsufficientMemory {
                need_gib: indicatif::HumanBytes(0),
            });
        }

        // Are we going to write out simulated auto-correlations? Use this
        // variable so the rest of the code is clearer.
        let using_autos = input_vis_params.using_autos
            && self
                .output_model_vis_params
                .as_ref()
                .map(|p| p.output_autos)
                .unwrap_or_default();

        // Get the time and frequency resolutions once; these functions issue
        // warnings if they have to guess, so doing this once means we aren't
        // issuing too many warnings.
        let obs_context = input_vis_params.get_obs_context();
        let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
        let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;

        let vis_shape = (
            selected_timeblocks.len(),
            input_vis_params.spw.chanblocks.len(),
            num_unflagged_cross_baselines,
        );
        let num_elems = vis_shape.0 * vis_shape.1 * vis_shape.2;
        // We need this many bytes for each of the data and model arrays to do
        // calibration.
        let size = indicatif::HumanBytes((num_elems * std::mem::size_of::<Jones<f32>>()) as u64);
        debug!("Shape of data and model arrays: ({} timesteps, {} channels, {} baselines; {size} each)", vis_shape.0, vis_shape.1, vis_shape.2);

        macro_rules! fallible_allocator {
            ($default:expr) => {{
                let mut v = Vec::new();
                match v.try_reserve_exact(num_elems) {
                    Ok(()) => {
                        v.resize(num_elems, $default);
                        Ok(Array3::from_shape_vec(vis_shape, v).unwrap())
                    }
                    Err(_) => {
                        // We need this many gibibytes to do calibration (two
                        // visibility arrays and one weights array).
                        let need_gib = indicatif::HumanBytes(
                            (num_elems
                                * (2 * std::mem::size_of::<Jones<f32>>()
                                    + std::mem::size_of::<f32>())) as u64,
                        );

                        Err(DiCalibrateError::InsufficientMemory {
                            // Instead of erroring out with how many bytes we need
                            // for the array we just tried to allocate, error out
                            // with how many bytes we need total.
                            need_gib,
                        })
                    }
                }
            }};
        }

        debug!("Allocating memory for input data visibilities and model visibilities");
        let cal_vis = expensive_op(
            || -> Result<_, DiCalibrateError> {
                let vis_data: Array3<Jones<f32>> = fallible_allocator!(Jones::default())?;
                let vis_model: Array3<Jones<f32>> = fallible_allocator!(Jones::default())?;
                let vis_weights: Array3<f32> = fallible_allocator!(0.0)?;
                Ok(CalVis {
                    vis_data,
                    vis_weights,
                    vis_model,
                    pols: Polarisations::default(),
                })
            },
            "Still waiting to allocate visibility memory",
        )?;
        let CalVis {
            mut vis_data,
            mut vis_model,
            mut vis_weights,
            pols: _,
        } = cal_vis;

        // Sky-modelling communication channel. Used to tell the model writer when
        // visibilities have been generated and they're ready to be written.
        let (tx_model, rx_model) = unbounded();

        // Progress bars. Courtesy Dev.
        let multi_progress = MultiProgress::with_draw_target(if PROGRESS_BARS.load() {
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        });
        let pb = ProgressBar::new(selected_timeblocks.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Reading data");
        let read_progress = multi_progress.add(pb);
        let pb = ProgressBar::new(selected_timeblocks.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Sky modelling");
        let model_progress = multi_progress.add(pb);
        // Only add a model writing progress bar if we need it.
        let model_write_progress = self.output_model_vis_params.as_ref().map(|o| {
            let pb = ProgressBar::new(o.output_timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Model writing");
            multi_progress.add(pb)
        });

        // Use a variable to track whether any threads have an issue.
        let error = AtomicCell::new(false);
        info!("Reading input data and sky modelling");
        thread::scope(|scope| -> Result<(), DiCalibrateError> {
            // Mutable slices of the "global" arrays. These allow threads to mutate
            // the global arrays in parallel (using the Arc<Mutex<_>> pattern would
            // kill performance here).
            let vis_data_slices = vis_data.outer_iter_mut();
            let vis_model_slices = vis_model.outer_iter_mut();
            let vis_weight_slices = vis_weights.outer_iter_mut();

            // Input visibility-data reading thread.
            let data_handle: ScopedJoinHandle<Result<(), VisReadError>> = thread::Builder::new()
                .name("read".to_string())
                .spawn_scoped(scope, || {
                    // If a panic happens, update our atomic error.
                    defer_on_unwind! { error.store(true); }
                    read_progress.tick();

                    for (timeblock, vis_data_fb, vis_weights_fb) in izip!(
                        selected_timeblocks,
                        vis_data_slices,
                        vis_weight_slices
                    ) {
                        let result = input_vis_params.read_timeblock(
                            timeblock,
                            vis_data_fb,
                            vis_weights_fb,
                            None,
                            &error,
                        );

                        // If the result of reading data was an error, allow the other
                        // threads to see this so they can abandon their work early.
                        if result.is_err() {
                            error.store(true);
                        }
                        result?;

                        // Should we continue?
                        if error.load() {
                            return Ok(());
                        }

                        read_progress.inc(1);
                    }

                    debug!("Finished reading");
                    read_progress.abandon_with_message("Finished reading visibilities");
                    Ok(())
                })
                .expect("OS can create threads");

            // Sky-model generation thread.
            let model_handle: ScopedJoinHandle<Result<(), ModelError>> = thread::Builder::new()
                .name("model".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }
                    model_progress.tick();

                    let result = model_thread(
                        &*self.beam,
                        &self.source_list,
                        input_vis_params,
                        &self.modelling_params,
                        selected_timeblocks,
                        using_autos,
                        vis_model_slices,
                        tx_model,
                        &error,
                        model_progress,
                    );
                    if result.is_err() {
                        error.store(true);
                    }
                    result
                })
                .expect("OS can create threads");

            // Model writing thread. If the user hasn't specified to write the model
            // to a file, then this thread just consumes messages from the modeller.
            let writer_handle: ScopedJoinHandle<Result<(), VisWriteError>> = thread::Builder::new()
                .name("model writer".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    // If the user wants the sky model written out,
                    // `output_model_vis_params` is populated.
                    if let Some(OutputVisParams {
                        output_files,
                        output_time_average_factor,
                        output_freq_average_factor,
                        output_autos: _,
                        output_timeblocks,
                        write_smallest_contiguous_band,
                    }) = &self.output_model_vis_params
                    {
                        if let Some(pb) = model_write_progress.as_ref() {
                            pb.tick();
                        }

                        // Form sorted unflagged tile pairs from our
                        // cross-correlation baselines (and maybe autos too).
                        let unflagged_baseline_tile_pairs: Vec<_> = if using_autos {
                            input_vis_params
                                .tile_baseline_flags
                                .get_unflagged_baseline_tile_pairs()
                                .collect()
                        } else {
                            input_vis_params
                                .tile_baseline_flags
                                .get_unflagged_cross_baseline_tile_pairs()
                                .collect()
                        };
                        let result = write_vis(
                            output_files,
                            obs_context.array_position,
                            obs_context.phase_centre,
                            obs_context.pointing_centre,
                            &obs_context.tile_xyzs,
                            &obs_context.tile_names,
                            obs_context.obsid,
                            output_timeblocks,
                            input_vis_params.time_res,
                            input_vis_params.dut1,
                            &input_vis_params.spw,
                            &unflagged_baseline_tile_pairs,
                            *output_time_average_factor,
                            *output_freq_average_factor,
                            input_vis_params.vis_reader.get_marlu_mwa_info().as_ref(),
                            *write_smallest_contiguous_band,
                            rx_model,
                            &error,
                            model_write_progress,
                        );
                        if result.is_err() {
                            error.store(true);
                        }
                        // Discard the result string.
                        result?;
                        Ok(())
                    } else {
                        // There's no model to write out, but we still need to handle all of the
                        // incoming messages.
                        for _ in rx_model.iter() {}
                        Ok(())
                    }
                })
                .expect("OS can create threads");

            // Join all thread handles. This propagates any errors and lets us know
            // if any threads panicked, if panics aren't aborting as per the
            // Cargo.toml. (It would be nice to capture the panic information, if
            // it's possible, but I don't know how, so panics are currently
            // aborting.)
            data_handle.join().unwrap()?;
            model_handle.join().unwrap()?;
            writer_handle.join().unwrap()?;
            Ok(())
        })?;

        let mut cal_vis = CalVis {
            vis_data,
            vis_weights,
            vis_model,
            pols: obs_context.polarisations,
        };
        cal_vis.scale_by_weights(Some(&self.baseline_weights));

        info!("Finished reading input data and sky modelling");

        Ok(cal_vis)
    }
}

#[allow(clippy::too_many_arguments)]
fn model_thread(
    beam: &dyn Beam,
    source_list: &SourceList,
    input_vis_params: &InputVisParams,
    modelling_params: &ModellingParams,
    selected_timeblocks: &[Timeblock],
    model_autos: bool,
    vis_model_slices: AxisIterMut<'_, Jones<f32>, Ix2>,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), ModelError> {
    let apply_precession = modelling_params.apply_precession;
    let obs_context = input_vis_params.get_obs_context();
    let unflagged_tile_xyzs = obs_context
        .tile_xyzs
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            !input_vis_params
                .tile_baseline_flags
                .flagged_tiles
                .contains(i)
        })
        .map(|(_, xyz)| *xyz)
        .collect::<Vec<_>>();
    let freqs = input_vis_params
        .spw
        .chanblocks
        .iter()
        .map(|c| c.freq)
        .collect::<Vec<_>>();
    let num_tiles = unflagged_tile_xyzs.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let auto_vis_shape = (freqs.len(), num_tiles);

    let weight_factor = ((input_vis_params.spw.freq_res / FREQ_WEIGHT_FACTOR)
        * (input_vis_params.time_res.to_seconds() / TIME_WEIGHT_FACTOR))
        as f32;

    #[cfg(any(feature = "cuda", feature = "hip"))]
    let use_baseline_shards = {
        use crate::{gpu::partition_baselines_across_devices, model::ModelDevice, MODEL_DEVICE};
        matches!(MODEL_DEVICE.load(), ModelDevice::Gpu)
            && modelling_params.gpu_devices.len() > 1
            && num_baselines > 0
            && partition_baselines_across_devices(num_baselines, &modelling_params.gpu_devices)
                .len()
                > 1
    };
    #[cfg(not(any(feature = "cuda", feature = "hip")))]
    let use_baseline_shards = false;

    if use_baseline_shards {
        #[cfg(any(feature = "cuda", feature = "hip"))]
        {
            return model_thread_baseline_sharded(
                beam,
                source_list,
                input_vis_params,
                modelling_params,
                selected_timeblocks,
                &unflagged_tile_xyzs,
                &freqs,
                model_autos,
                vis_model_slices,
                tx,
                error,
                progress_bar,
                weight_factor,
                auto_vis_shape,
            );
        }
    }

    let modeller = new_sky_modeller(
        beam,
        source_list,
        obs_context.polarisations,
        &unflagged_tile_xyzs,
        &freqs,
        &input_vis_params.tile_baseline_flags.flagged_tiles,
        obs_context.phase_centre,
        obs_context.array_position.longitude_rad,
        obs_context.array_position.latitude_rad,
        input_vis_params.dut1,
        apply_precession,
    )?;

    // Iterate over all calibration timesteps and write to the model slices.
    for (timestamp, mut vis_model_fb) in selected_timeblocks
        .iter()
        .map(|tb| tb.median)
        .zip(vis_model_slices)
    {
        debug!("Modelling timestamp {}", timestamp.to_gpst_seconds());
        modeller.model_timestep_with(timestamp, vis_model_fb.view_mut())?;
        let auto_data_fb = if model_autos {
            let mut auto_data_fb = ArcArray2::zeros(auto_vis_shape);
            modeller.model_timestep_autos_with(timestamp, auto_data_fb.view_mut())?;
            Some(auto_data_fb)
        } else {
            None
        };

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data_fb: vis_model_fb.to_shared(),
            cross_weights_fb: ArcArray::from_elem(vis_model_fb.dim(), weight_factor),
            autos: auto_data_fb.map(|d| (d, ArcArray2::from_elem(auto_vis_shape, weight_factor))),
            timestamp,
        }) {
            Ok(()) => (),
            // If we can't send the message, it's because the channel has
            // been closed on the other side. That should only happen
            // because the writer has exited due to error; in that case,
            // just exit this thread.
            Err(_) => return Ok(()),
        }
        progress_bar.inc(1);
    }

    debug!("Finished modelling");
    progress_bar.abandon_with_message("Finished generating sky model");
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "hip"))]
#[allow(clippy::too_many_arguments)]
fn model_thread_baseline_sharded(
    beam: &dyn Beam,
    source_list: &SourceList,
    input_vis_params: &InputVisParams,
    modelling_params: &ModellingParams,
    selected_timeblocks: &[Timeblock],
    unflagged_tile_xyzs: &[marlu::XyzGeodetic],
    freqs: &[f64],
    model_autos: bool,
    vis_model_slices: AxisIterMut<'_, Jones<f32>, Ix2>,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
    weight_factor: f32,
    auto_vis_shape: (usize, usize),
) -> Result<(), ModelError> {
    use crate::{
        gpu::partition_baselines_across_devices,
        model::SkyModellerGpu,
    };

    let obs_context = input_vis_params.get_obs_context();
    let apply_precession = modelling_params.apply_precession;
    let num_baselines = (unflagged_tile_xyzs.len() * (unflagged_tile_xyzs.len() - 1)) / 2;
    let shards =
        partition_baselines_across_devices(num_baselines, &modelling_params.gpu_devices);
    info!(
        "Baseline-sharded GPU modelling across {} device(s) ({} baselines)",
        shards.len(),
        num_baselines
    );

    let mut modellers = shards
        .iter()
        .map(|&(device, offset, count)| {
            SkyModellerGpu::new_on_device(
                beam,
                source_list,
                obs_context.polarisations,
                unflagged_tile_xyzs,
                freqs,
                &input_vis_params.tile_baseline_flags.flagged_tiles,
                obs_context.phase_centre,
                obs_context.array_position.longitude_rad,
                obs_context.array_position.latitude_rad,
                input_vis_params.dut1,
                apply_precession,
                device,
                offset,
                Some(count),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Autos still need a full-array modeller (tile axis, not baselines).
    let auto_modeller = if model_autos {
        Some(new_sky_modeller(
            beam,
            source_list,
            obs_context.polarisations,
            unflagged_tile_xyzs,
            freqs,
            &input_vis_params.tile_baseline_flags.flagged_tiles,
            obs_context.phase_centre,
            obs_context.array_position.longitude_rad,
            obs_context.array_position.latitude_rad,
            input_vis_params.dut1,
            apply_precession,
        )?)
    } else {
        None
    };

    let num_freqs = freqs.len();
    for (timestamp, mut vis_model_fb) in selected_timeblocks
        .iter()
        .map(|tb| tb.median)
        .zip(vis_model_slices)
    {
        debug!(
            "Modelling timestamp {} with baseline shards",
            timestamp.to_gpst_seconds()
        );

        let pieces: Result<Vec<_>, ModelError> = modellers
            .par_iter_mut()
            .zip(shards.par_iter())
            .map(|(modeller, &(device, offset, count))| {
                crate::gpu::set_device(device)?;
                let mut shard = Array2::zeros((num_freqs, count));
                SkyModeller::model_timestep_with(modeller, timestamp, shard.view_mut())?;
                Ok((offset, shard))
            })
            .collect();
        for (offset, shard) in pieces? {
            let end = offset + shard.len_of(Axis(1));
            vis_model_fb
                .slice_mut(s![.., offset..end])
                .assign(&shard);
        }

        let auto_data_fb = if let Some(auto_modeller) = auto_modeller.as_ref() {
            let mut auto_data_fb = ArcArray2::zeros(auto_vis_shape);
            auto_modeller.model_timestep_autos_with(timestamp, auto_data_fb.view_mut())?;
            Some(auto_data_fb)
        } else {
            None
        };

        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data_fb: vis_model_fb.to_shared(),
            cross_weights_fb: ArcArray::from_elem(vis_model_fb.dim(), weight_factor),
            autos: auto_data_fb.map(|d| (d, ArcArray2::from_elem(auto_vis_shape, weight_factor))),
            timestamp,
        }) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        }
        progress_bar.inc(1);
    }

    debug!("Finished modelling");
    progress_bar.abandon_with_message("Finished generating sky model");
    Ok(())
}

pub(crate) struct CalVis {
    /// Visibilites read from input data.
    pub(crate) vis_data: Array3<Jones<f32>>,

    /// The weights on the visibilites read from input data.
    pub(crate) vis_weights: Array3<f32>,

    /// Visibilites generated from the sky-model source list.
    pub(crate) vis_model: Array3<Jones<f32>>,

    /// The available polarisations within the data.
    pub(crate) pols: Polarisations,
}

impl CalVis {
    // Multiply the data and model visibilities by the weights (and baseline
    // weights that could be e.g. based on UVW cuts). If a weight is negative,
    // it means the corresponding visibility should be flagged, so that
    // visibility is set to 0; this means it does not affect calibration. Not
    // iterating over weights during calibration makes makes calibration run
    // significantly faster.
    pub(crate) fn scale_by_weights(&mut self, baseline_weights: Option<&[f64]>) {
        debug!("Multiplying visibilities by weights");

        // Ensure that the number of baseline weights is the same as the number
        // of baselines.
        if let Some(w) = baseline_weights {
            assert_eq!(w.len(), self.vis_data.len_of(Axis(2)));
        }

        self.vis_data
            .outer_iter_mut()
            .into_par_iter()
            .zip(self.vis_model.outer_iter_mut())
            .zip(self.vis_weights.outer_iter())
            .for_each(|((mut vis_data, mut vis_model), vis_weights)| {
                vis_data
                    .outer_iter_mut()
                    .zip(vis_model.outer_iter_mut())
                    .zip(vis_weights.outer_iter())
                    .for_each(|((mut vis_data, mut vis_model), vis_weights)| {
                        vis_data
                            .iter_mut()
                            .zip(vis_model.iter_mut())
                            .zip(vis_weights.iter())
                            .zip(
                                baseline_weights
                                    .map(|w| w.iter().cycle())
                                    .unwrap_or_else(|| [1.0].iter().cycle()),
                            )
                            .for_each(
                                |(((vis_data, vis_model), &vis_weight), &baseline_weight)| {
                                    let weight = f64::from(vis_weight) * baseline_weight;
                                    if weight <= 0.0 {
                                        *vis_data = Jones::default();
                                        *vis_model = Jones::default();
                                    } else {
                                        *vis_data = Jones::<f32>::from(
                                            Jones::<f64>::from(*vis_data) * weight,
                                        );
                                        *vis_model = Jones::<f32>::from(
                                            Jones::<f64>::from(*vis_model) * weight,
                                        );
                                    }
                                },
                            );
                    });
            });
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum DiCalibrateError {
    #[error("Insufficient memory available to perform calibration; need {need_gib} of memory.\nYou could try using fewer timesteps and channels.")]
    InsufficientMemory { need_gib: indicatif::HumanBytes },

    #[error(transparent)]
    SolutionsRead(#[from] crate::solutions::SolutionsReadError),

    #[error(transparent)]
    SolutionsWrite(#[from] crate::solutions::SolutionsWriteError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    VisRead(#[from] crate::io::read::VisReadError),

    #[error(transparent)]
    VisWrite(#[from] crate::io::write::VisWriteError),

    #[error(transparent)]
    Fitsio(#[from] fitsio::errors::Error),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
