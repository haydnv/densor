use std::sync::Arc;

use ocl::core::{DeviceInfo, DeviceInfoResult};
use ocl::{Buffer, Context, Device, DeviceType, Kernel, Platform, Queue};
use rayon::prelude::*;

use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
use crate::ops::{
    Construct, ElementwiseCompare, ElementwiseDual, ElementwiseUnary, Random, Reduce, Transform,
};
use crate::platform::{Convert, PlatformInstance};
use crate::{strides_for, CType, Error, Float, Range, Shape};

use super::ops::*;
use super::{programs, CLConverter};
use super::{CL_PLATFORM, WG_SIZE};

pub const GPU_MIN_SIZE: usize = 1024; // 1 KiB

pub const ACC_MIN_SIZE: usize = 2_147_483_648; // 1 GiB

#[derive(Clone)]
struct DeviceList {
    devices: Vec<Device>,
    next: Arc<std::sync::atomic::AtomicUsize>,
}

impl Default for DeviceList {
    fn default() -> Self {
        Self {
            devices: Vec::default(),
            next: Arc::new(Default::default()),
        }
    }
}

impl DeviceList {
    fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    fn next(&self) -> Option<Device> {
        if self.devices.is_empty() {
            None
        } else {
            let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.devices.get(idx % self.devices.len()).copied()
        }
    }
}

impl From<Vec<Device>> for DeviceList {
    fn from(devices: Vec<Device>) -> Self {
        Self {
            devices,
            next: Arc::new(Default::default()),
        }
    }
}

impl FromIterator<Device> for DeviceList {
    fn from_iter<T: IntoIterator<Item = Device>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<Device>>())
    }
}

pub struct CLPlatform {
    cl_platform: Platform,
    cl_context: Context,
    cl_cpus: DeviceList,
    cl_gpus: DeviceList,
    cl_accs: DeviceList,
}

impl CLPlatform {
    pub(super) fn default() -> Result<Self, Error> {
        let cl_platform = Platform::first()?;
        Self::try_from(cl_platform).map_err(Error::from)
    }

    fn next_cpu(&self) -> Option<Device> {
        self.cl_cpus.next()
    }

    fn next_gpu(&self) -> Option<Device> {
        self.cl_gpus.next()
    }

    fn next_acc(&self) -> Option<Device> {
        self.cl_accs.next()
    }

    fn select_device_type(&self, size_hint: usize) -> DeviceType {
        if size_hint < GPU_MIN_SIZE {
            DeviceType::CPU
        } else if size_hint < ACC_MIN_SIZE {
            DeviceType::GPU
        } else {
            DeviceType::ACCELERATOR
        }
    }

    fn select_device(&self, device_type: DeviceType) -> Option<Device> {
        match device_type {
            DeviceType::CPU => self
                .next_cpu()
                .or_else(|| self.next_gpu())
                .or_else(|| self.next_acc()),

            DeviceType::GPU => self
                .next_gpu()
                .or_else(|| self.next_acc())
                .or_else(|| self.next_cpu()),

            DeviceType::ACCELERATOR => self
                .next_acc()
                .or_else(|| self.next_gpu())
                .or_else(|| self.next_cpu()),

            other => panic!("unsupported OpenCL device type: {other:?}"),
        }
    }
}

impl TryFrom<Platform> for CLPlatform {
    type Error = ocl::Error;

    fn try_from(cl_platform: Platform) -> Result<Self, Self::Error> {
        let devices = Device::list(cl_platform, None)?;
        let cl_context = ocl::builders::ContextBuilder::new()
            .platform(cl_platform)
            .devices(&devices)
            .build()?;

        let cl_cpus = Device::list(cl_platform, Some(DeviceType::CPU))?;
        let cl_gpus = Device::list(cl_platform, Some(DeviceType::GPU))?;
        let cl_accs = Device::list(cl_platform, Some(DeviceType::ACCELERATOR))?;

        Ok(Self {
            cl_cpus: cl_cpus.into(),
            cl_gpus: cl_gpus.into(),
            cl_accs: cl_accs.into(),
            cl_context,
            cl_platform,
        })
    }
}

/// The OpenCL platform
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct OpenCL;

impl PlatformInstance for OpenCL {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl OpenCL {
    /// Borrow the OpenCL [`Context`] of this platform.
    pub fn context<'a>() -> &'a Context {
        &CL_PLATFORM.cl_context
    }

    /// Copy the given `data` into a new [`Buffer`].
    pub fn copy_into_buffer<T: CType>(data: &[T]) -> Result<Buffer<T>, ocl::Error> {
        ocl::builders::BufferBuilder::new()
            .len(data.len())
            .context(&CL_PLATFORM.cl_context)
            .copy_host_slice(data)
            .build()
    }

    /// Create a new [`Buffer`].
    pub fn create_buffer<T: CType>(size: usize) -> Result<Buffer<T>, ocl::Error> {
        ocl::builders::BufferBuilder::new()
            .len(size)
            .context(&CL_PLATFORM.cl_context)
            .build()
    }

    pub(crate) fn queue(
        size_hint: usize,
        left: Option<&Queue>,
        right: Option<&Queue>,
    ) -> Result<Queue, ocl::Error> {
        let device_type = CL_PLATFORM.select_device_type(size_hint);

        // TODO: is this slow?
        if let Some(queue) = left {
            if let DeviceInfoResult::Type(dt) = queue.device().info(DeviceInfo::Type)? {
                if dt == device_type {
                    return Ok(queue.clone());
                }
            }
        }

        // TODO: is this slow?
        if let Some(queue) = right {
            if let DeviceInfoResult::Type(dt) = queue.device().info(DeviceInfo::Type)? {
                if dt == device_type {
                    return Ok(queue.clone());
                }
            }
        }

        let device = CL_PLATFORM
            .select_device(device_type)
            .expect("OpenCL device");

        Queue::new(&CL_PLATFORM.cl_context, device, None)
    }
}

impl<'a, T: CType> Convert<'a, T> for OpenCL {
    type Buffer = CLConverter<'a, T>;

    fn convert(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error> {
        buffer.to_cl().map_err(Error::from)
    }
}

impl<T: CType> Construct<T> for OpenCL {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        if start <= stop {
            let step = (stop - start).to_float().to_f64() / size as f64;
            Linear::new(start, step, size).map(AccessOp::from)
        } else {
            Err(Error::Bounds(format!("invalid range: [{start}, {stop})")))
        }
    }
}

impl<T, L, R> ElementwiseCompare<L, R, T> for OpenCL
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Compare::eq(left, right).map(AccessOp::from)
    }
}

impl<T, L, R> ElementwiseDual<L, R, T> for OpenCL
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::add(left, right).map(AccessOp::from)
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::sub(left, right).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: CType> ElementwiseUnary<A, T> for OpenCL {
    type Op = Unary<A, T, T>;

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::ln(access).map(AccessOp::from)
    }
}

impl Random for OpenCL {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        RandomNormal::new(size).map(AccessOp::from)
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        RandomUniform::new(size).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: CType> Reduce<A, T> for OpenCL {
    fn all(self, access: A) -> Result<bool, Error> {
        let buffer = access.read()?.to_cl()?;

        let result = [1];

        let program = programs::reduce::all(T::TYPE)?;

        let flag = unsafe {
            Buffer::builder()
                .context(Self::context())
                .use_host_slice(&result)
                .len(1)
                .build()?
        };

        let queue = Self::queue(buffer.len(), buffer.default_queue(), None)?;

        let kernel = Kernel::builder()
            .name("all")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(buffer.len())
            .arg(&flag)
            .arg(&*buffer)
            .build()?;

        unsafe { kernel.enq()? }

        queue.finish()?;

        Ok(result == [1])
    }

    fn any(self, access: A) -> Result<bool, Error> {
        let buffer = access.read()?.to_cl()?;

        let result = [0];

        let program = programs::reduce::any(T::TYPE)?;

        let flag = unsafe {
            Buffer::builder()
                .context(Self::context())
                .use_host_slice(&result)
                .len(1)
                .build()?
        };

        let queue = Self::queue(buffer.len(), buffer.default_queue(), None)?;

        let kernel = Kernel::builder()
            .name("any")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(buffer.len())
            .arg(&flag)
            .arg(&*buffer)
            .build()?;

        unsafe { kernel.enq()? }

        queue.finish()?;

        Ok(result == [1])
    }

    fn sum(self, access: A) -> Result<T, Error> {
        const MIN_SIZE: usize = 8192;

        let input = access.read()?.to_cl()?;

        let min_size = MIN_SIZE * num_cpus::get();

        if input.len() < min_size {
            let mut result = vec![T::ZERO; input.len()];
            input.read(&mut result).enq()?;
            return Ok(result.into_par_iter().sum());
        }

        let queue = Self::queue(input.size(), input.default_queue(), None)?;

        let program = programs::reduce::reduce(T::TYPE, "add")?;

        let mut buffer = {
            let output = Buffer::builder()
                .queue(queue.clone())
                .len(input.len().div_ceil(WG_SIZE))
                .fill_val(T::ZERO)
                .build()?;

            let kernel = Kernel::builder()
                .name("reduce")
                .program(&program)
                .queue(queue.clone())
                .local_work_size(WG_SIZE)
                .global_work_size(WG_SIZE * output.len())
                .arg(input.len() as u64)
                .arg(&*input)
                .arg(&output)
                .arg_local::<T>(WG_SIZE)
                .build()?;

            unsafe { kernel.enq()? };

            output
        };

        while buffer.len() >= min_size {
            let input = buffer;

            let output = Buffer::builder()
                .queue(queue.clone())
                .len(input.len().div_ceil(WG_SIZE))
                .fill_val(T::ZERO)
                .build()?;

            let kernel = Kernel::builder()
                .name("reduce")
                .program(&program)
                .queue(queue.clone())
                .local_work_size(WG_SIZE)
                .global_work_size(WG_SIZE * output.len())
                .arg(input.len() as u64)
                .arg(&input)
                .arg(&output)
                .arg_local::<T>(WG_SIZE)
                .build()?;

            unsafe { kernel.enq()? }

            buffer = output;
        }

        let mut result = vec![T::ZERO; buffer.len()];
        buffer.read(&mut result).enq()?;

        queue.finish()?;

        Ok(result.into_par_iter().sum())
    }
}

impl<A, T> Transform<A, T> for OpenCL
where
    A: Access<T>,
    T: CType,
{
    type Broadcast = View<A, T>;
    type Slice = Slice<A, T>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        let strides = strides_for(&shape, broadcast.len()).collect();
        View::new(access, shape, broadcast, strides).map(AccessOp::from)
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        Slice::new(access, shape, range).map(AccessOp::from)
    }
}
