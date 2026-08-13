// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Utilities for CUDA/HIP devices.
//!
//! We assume that everything is UTF-8.

include!("utils_bindings.rs");

use std::{
    ffi::{CStr, CString},
    panic::Location,
};

use super::GpuError;

#[derive(Debug, Clone)]
pub(crate) struct GpuDriverInfo {
    /// Formatted CUDA/HIP driver version, e.g. "11.7".
    pub(crate) driver_version: Box<str>,
    /// Formatted CUDA/HIP runtime version, e.g. "11.7".
    pub(crate) runtime_version: Box<str>,
}

#[derive(Debug, Clone)]
pub(crate) struct GpuDeviceInfo {
    pub(crate) name: Box<str>,
    pub(crate) capability: Box<str>,
    /// \[MebiBytes (MiB)\]
    pub(crate) total_global_mem: usize,
}

fn gpu_c_error(error_message_ptr: *const std::os::raw::c_char) -> GpuError {
    let error_message = unsafe { CStr::from_ptr(error_message_ptr).to_str() };
    #[cfg(feature = "cuda")]
    let error_message = error_message.unwrap_or("<cannot read CUDA error string>");
    #[cfg(feature = "hip")]
    let error_message = error_message.unwrap_or("<cannot read HIP error string>");
    let location = Location::caller();
    GpuError::Generic {
        msg: error_message.into(),
        file: location.file(),
        line: location.line(),
    }
}

/// Number of CUDA/HIP devices visible to this process.
pub(crate) fn get_device_count() -> Result<i32, GpuError> {
    unsafe {
        let mut count = 0;
        let error_message_ptr = get_gpu_device_count(&mut count);
        if !error_message_ptr.is_null() {
            return Err(gpu_c_error(error_message_ptr));
        }
        Ok(count)
    }
}

/// Set the calling thread's current CUDA/HIP device.
pub(crate) fn set_device(device: i32) -> Result<(), GpuError> {
    unsafe {
        let error_message_ptr = set_gpu_device(device);
        if !error_message_ptr.is_null() {
            return Err(gpu_c_error(error_message_ptr));
        }
        Ok(())
    }
}

/// Get CUDA/HIP device and driver information for `device`.
pub(crate) fn get_device_info(device: i32) -> Result<(GpuDeviceInfo, GpuDriverInfo), GpuError> {
    unsafe {
        let name = CString::from_vec_unchecked(vec![1; 256]).into_raw();
        let mut device_major = 0;
        let mut device_minor = 0;
        let mut total_global_mem = 0;
        let mut driver_version = 0;
        let mut runtime_version = 0;
        let error_message_ptr = get_gpu_device_info(
            device,
            name,
            &mut device_major,
            &mut device_minor,
            &mut total_global_mem,
            &mut driver_version,
            &mut runtime_version,
        );
        if !error_message_ptr.is_null() {
            return Err(gpu_c_error(error_message_ptr));
        }

        let device_info = GpuDeviceInfo {
            name: CString::from_raw(name)
                .to_str()
                .expect("GPU device name isn't UTF-8")
                .to_string()
                .into_boxed_str(),
            capability: format!("{device_major}.{device_minor}").into_boxed_str(),
            total_global_mem: total_global_mem / 1048576,
        };

        #[cfg(feature = "cuda")]
        let (driver_version, runtime_version) = {
            let d = format!("{}.{}", driver_version / 1000, (driver_version / 10) % 100);
            let r = format!(
                "{}.{}",
                runtime_version / 1000,
                (runtime_version / 10) % 100
            );
            (d, r)
        };
        #[cfg(feature = "hip")]
        let (driver_version, runtime_version) = {
            // This isn't documented, but is the only thing that makes sense to
            // me.
            let d = format!(
                "{}.{}",
                driver_version / 10_000_000,
                (driver_version / 10_000) % 100
            );
            let r = format!(
                "{}.{}",
                runtime_version / 10_000_000,
                (runtime_version / 10_000) % 100
            );
            (d, r)
        };

        Ok((
            device_info,
            GpuDriverInfo {
                driver_version: driver_version.into_boxed_str(),
                runtime_version: runtime_version.into_boxed_str(),
            },
        ))
    }
}

/// Partition `num_baselines` into contiguous shards, one per device.
///
/// Returns `(device_id, baseline_offset, baseline_count)` triples. Empty input
/// yields an empty vec. Extra devices beyond `num_baselines` are unused.
pub(crate) fn partition_baselines_across_devices(
    num_baselines: usize,
    devices: &[i32],
) -> Vec<(i32, usize, usize)> {
    if num_baselines == 0 || devices.is_empty() {
        return vec![];
    }
    let n_shards = devices.len().min(num_baselines);
    let base = num_baselines / n_shards;
    let rem = num_baselines % n_shards;
    let mut offset = 0;
    let mut out = Vec::with_capacity(n_shards);
    for (i, &device) in devices.iter().take(n_shards).enumerate() {
        let count = base + usize::from(i < rem);
        out.push((device, offset, count));
        offset += count;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_baselines_even() {
        let shards = partition_baselines_across_devices(10, &[0, 1]);
        assert_eq!(shards, vec![(0, 0, 5), (1, 5, 5)]);
    }

    #[test]
    fn partition_baselines_uneven() {
        let shards = partition_baselines_across_devices(11, &[0, 1, 2]);
        assert_eq!(shards, vec![(0, 0, 4), (1, 4, 4), (2, 8, 3)]);
    }

    #[test]
    fn partition_more_devices_than_baselines() {
        let shards = partition_baselines_across_devices(2, &[0, 1, 2, 3]);
        assert_eq!(shards, vec![(0, 0, 1), (1, 1, 1)]);
    }
}
