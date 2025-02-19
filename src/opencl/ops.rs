use std::borrow::BorrowMut;
use std::fmt;
use std::marker::PhantomData;

use frand::Rand;
use ocl::{Buffer, Kernel, Program, Queue};

use crate::access::{Access, AccessBuf, AccessMut};
use crate::ops::{Enqueue, Op, ReadValue, ReduceAll, SliceSpec, ViewSpec, Write};
use crate::{strides_for, Axes, BufferConverter, CType, Error, Float, Range, Shape, Strides};

use super::platform::OpenCL;
use super::{programs, TILE_SIZE, WG_SIZE};

pub struct Cast<A, IT, OT> {
    access: A,
    program: Program,
    dtype: PhantomData<(IT, OT)>,
}

impl<A, IT: CType, OT: CType> Cast<A, IT, OT> {
    pub fn new(access: A) -> Result<Self, Error> {
        programs::elementwise::cast(IT::TYPE, OT::TYPE).map(|program| Self {
            access,
            program,
            dtype: PhantomData,
        })
    }
}

impl<A: Access<IT>, IT: CType, OT: CType> Op for Cast<A, IT, OT> {
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A: Access<IT>, IT: CType, OT: CType> Enqueue<OpenCL, OT> for Cast<A, IT, OT> {
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;
        let queue = OpenCL::queue(input.len(), &[input.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("cast")
            .program(&self.program)
            .queue(queue)
            .global_work_size(input.len())
            .arg(&*input)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }
}

impl<A: Access<IT>, IT: CType, OT: CType> ReadValue<OpenCL, OT> for Cast<A, IT, OT> {
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access
            .read_value(offset)
            .map(|n| n.to_f64())
            .map(OT::from_f64)
    }
}

pub struct Dual<L, R, IT, OT> {
    left: L,
    right: R,
    program: Program,
    op: fn(IT, IT) -> OT,
}

impl<L, R, IT, OT> Dual<L, R, IT, OT> {
    fn new(left: L, right: R, program: Program, op: fn(IT, IT) -> OT) -> Result<Self, Error> {
        Ok(Self {
            left,
            right,
            program,
            op,
        })
    }
}

// arithmetic
impl<L, R, T: CType> Dual<L, R, T, T> {
    pub fn add(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "add")?;
        Self::new(left, right, program, T::add)
    }

    pub fn div(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "div")?;
        Self::new(left, right, program, T::div)
    }

    pub fn log(arg: L, exp: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "_log")?;
        Self::new(arg, exp, program, |a, e| {
            T::from_float(a.to_float().log(e.to_float()))
        })
    }

    pub fn mul(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "mul")?;
        Self::new(left, right, program, T::mul)
    }

    pub fn pow(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "pow")?;
        Self::new(left, right, program, T::pow)
    }

    pub fn rem(left: L, right: R) -> Result<Self, Error> {
        let program = if T::IS_FLOAT { "fmod" } else { "mod" };
        let program = programs::elementwise::dual(T::TYPE, program)?;
        Self::new(left, right, program, T::rem)
    }

    pub fn sub(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "sub")?;
        Self::new(left, right, program, T::sub)
    }
}

// boolean operations
impl<L, R, T: CType> Dual<L, R, T, u8> {
    pub fn and(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "and")?;
        let op = |l, r| if l != T::ZERO && r != T::ZERO { 1 } else { 0 };
        Self::new(left, right, program, op)
    }

    pub fn or(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "or")?;
        let op = |l, r| if l != T::ZERO || r != T::ZERO { 1 } else { 0 };
        Self::new(left, right, program, op)
    }

    pub fn xor(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "xor")?;
        let op = |l, r| {
            if (l != T::ZERO) ^ (r != T::ZERO) {
                1
            } else {
                0
            }
        };
        Self::new(left, right, program, op)
    }
}

// comparison
impl<L, R, T: CType> Dual<L, R, T, u8> {
    pub fn eq(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "eq")?;
        let op = |l, r| if l == r { 1 } else { 0 };
        Self::new(left, right, program, op)
    }

    pub fn ge(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "ge")?;
        let op = |l, r| if l >= r { 1 } else { 0 };
        Self::new(left, right, program, op)
    }

    pub fn gt(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "gt")?;
        let op = |l, r| if l > r { 1 } else { 0 };
        Self::new(left, right, program, op)
    }

    pub fn le(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "le")?;
        let op = |l, r| if l <= r { 1 } else { 0 };
        Self::new(left, right, program, op)
    }

    pub fn lt(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "lt")?;
        let op = |l, r| if l < r { 1 } else { 0 };
        Self::new(left, right, program, op)
    }

    pub fn ne(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual_boolean(T::TYPE, "ne")?;
        let op = |l, r| if l != r { 1 } else { 0 };
        Self::new(left, right, program, op)
    }
}

impl<L, R, IT, OT> Op for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, IT, OT> Enqueue<OpenCL, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_cl()?;
        let right = self.right.read()?.to_cl()?;
        debug_assert_eq!(left.len(), right.len());

        let queue = OpenCL::queue(left.len(), &[left.default_queue(), right.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("dual")
            .program(&self.program)
            .queue(queue)
            .global_work_size(left.len())
            .arg(&*left)
            .arg(&*right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<L, R, IT, OT> ReadValue<OpenCL, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        let l = self.left.read_value(offset)?;
        let r = self.right.read_value(offset)?;
        Ok((self.op)(l, r))
    }
}

pub struct Cond<A, L, R, T> {
    cond: A,
    then: L,
    or_else: R,
    program: Program,
    dtype: PhantomData<T>,
}

impl<A, L, R, T> Cond<A, L, R, T>
where
    T: CType,
{
    pub fn new(cond: A, then: L, or_else: R) -> Result<Self, Error> {
        let program = programs::gather::gather_cond(T::TYPE)?;

        Ok(Self {
            cond,
            then,
            or_else,
            program,
            dtype: PhantomData,
        })
    }
}

impl<A, L, R, T> Op for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        debug_assert_eq!(self.cond.size(), self.then.size());
        debug_assert_eq!(self.cond.size(), self.or_else.size());
        self.cond.size()
    }
}

impl<A, L, R, T> Enqueue<OpenCL, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let cond = self.cond.read()?;
        let then = self.then.read()?;
        let or_else = self.or_else.read()?;

        debug_assert_eq!(cond.len(), then.len());
        debug_assert_eq!(cond.len(), or_else.len());

        let (cond, (then, or_else)) = (cond.to_cl()?, (then.to_cl()?, or_else.to_cl()?));

        let queue = OpenCL::queue(
            cond.len(),
            &[
                cond.default_queue(),
                then.default_queue(),
                or_else.default_queue(),
            ],
        )?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(cond.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("gather_cond")
            .queue(queue)
            .program(&self.program)
            .global_work_size(cond.len())
            .arg(&*cond)
            .arg(&*then)
            .arg(&*or_else)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A, L, R, T> ReadValue<OpenCL, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let cond = self.cond.read_value(offset)?;
        let then = self.then.read_value(offset)?;
        let or_else = self.or_else.read_value(offset)?;

        if cond != 0 {
            Ok(then)
        } else {
            Ok(or_else)
        }
    }
}

pub struct MatDiag<A, T> {
    access: A,
    dim: usize,
    batch_size: usize,
    program: Program,
    dtype: PhantomData<T>,
}

impl<A, T: CType> MatDiag<A, T> {
    pub fn new(access: A, batch_size: usize, dim: usize) -> Result<Self, Error> {
        let program = programs::linalg::diagonal(T::TYPE)?;

        Ok(Self {
            access,
            batch_size,
            dim,
            program,
            dtype: PhantomData,
        })
    }
}

impl<A: Access<T>, T: CType> Op for MatDiag<A, T> {
    fn size(&self) -> usize {
        debug_assert_eq!(self.access.size(), self.batch_size * self.dim * self.dim);
        self.batch_size * self.dim
    }
}

impl<A: Access<T>, T: CType> Enqueue<OpenCL, T> for MatDiag<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;

        debug_assert_eq!(input.len(), self.batch_size * self.dim * self.dim);

        let queue = OpenCL::queue(self.size(), &[input.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.batch_size * self.dim)
            .build()?;

        let kernel = Kernel::builder()
            .name("diagonal")
            .program(&self.program)
            .queue(queue)
            .global_work_size((self.batch_size, self.dim))
            .arg(&*input)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }
}

impl<A: Access<T>, T: CType> ReadValue<OpenCL, T> for MatDiag<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let batch = offset / self.batch_size;
        let i = offset % self.batch_size;
        let source_offset = (batch * self.dim * self.dim) + (i * self.dim) + i;
        self.access.read_value(source_offset)
    }
}

pub struct MatMul<L, R, T> {
    left: L,
    right: R,
    batch_size: usize,
    dims: [usize; 3],
    padded: [usize; 3],
    pad_matrices: Program,
    matmul: Program,
    dtype: PhantomData<T>,
}

impl<L, R, T> MatMul<L, R, T>
where
    T: CType,
{
    pub fn new(left: L, right: R, dims: [usize; 4]) -> Result<Self, Error> {
        let pad_matrices = programs::linalg::pad_matrices(T::TYPE)?;
        let matmul = programs::linalg::matmul(T::TYPE)?;

        let [batch_size, a, b, c] = dims;
        assert!(batch_size > 0);

        let dims = [a, b, c];

        let padded = [
            pad_dim(a, TILE_SIZE),
            pad_dim(b, TILE_SIZE),
            pad_dim(c, TILE_SIZE),
        ];

        Ok(Self {
            left,
            right,
            batch_size,
            dims,
            padded,
            pad_matrices,
            matmul,
            dtype: PhantomData,
        })
    }

    fn matmul(
        &self,
        left: &Buffer<T>,
        right: &Buffer<T>,
        dims: [usize; 3],
    ) -> Result<Buffer<T>, Error> {
        let [a, b, c] = dims;

        assert_eq!(self.batch_size * a * b, left.len());
        assert_eq!(self.batch_size * b * c, right.len());

        let queue = OpenCL::queue(
            self.batch_size * a * b * c,
            &[left.default_queue(), right.default_queue()],
        )?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.batch_size * a * c)
            .fill_val(T::ZERO)
            .build()?;

        let dim4 = [a as u64, b as u64, c as u64, self.batch_size as u64];

        let kernel = Kernel::builder()
            .name("matmul")
            .program(&self.matmul)
            .queue(queue)
            .global_work_size((
                self.batch_size,
                a.div_ceil(TILE_SIZE),
                c.div_ceil(TILE_SIZE),
            ))
            .arg(ocl::core::Ulong4::from(dim4))
            .arg(b.div_ceil(TILE_SIZE))
            .arg(left)
            .arg(right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn pad_matrices<'a>(
        &self,
        batch: &Buffer<T>,
        dims_in: [usize; 2],
        dims_out: [usize; 2],
    ) -> Result<Buffer<T>, Error> {
        if dims_in == dims_out {
            return Ok(batch.clone());
        }

        assert_eq!(batch.len(), self.batch_size * dims_in[0] * dims_in[1]);

        let queue = OpenCL::queue(batch.len(), &[batch.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.batch_size * dims_out[0] * dims_out[1])
            .fill_val(T::ZERO)
            .build()?;

        let gws = if dims_in.iter().product::<usize>() <= dims_out.iter().product::<usize>() {
            (self.batch_size, dims_in[0], dims_in[1])
        } else {
            (self.batch_size, dims_out[0], dims_out[1])
        };

        let strides_in = [(dims_in[0] * dims_in[1]) as u64, dims_in[1] as u64];

        let strides_out = [(dims_out[0] * dims_out[1]) as u64, dims_out[1] as u64];

        let kernel = Kernel::builder()
            .name("pad_matrices")
            .program(&self.pad_matrices)
            .queue(queue)
            .global_work_size(gws)
            .arg(ocl::core::Ulong2::from(strides_in))
            .arg(ocl::core::Ulong2::from(strides_out))
            .arg(batch)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output.into())
    }
}

impl<L, R, T> Op for MatMul<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: Send + Sync,
{
    fn size(&self) -> usize {
        self.batch_size * self.dims[0] * self.dims[2]
    }
}

impl<L, R, T> Enqueue<OpenCL, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let [a, b, c] = self.dims;
        let [a_pad, b_pad, c_pad] = self.padded;

        let left = self.left.read()?.to_cl()?;
        let right = self.right.read()?.to_cl()?;

        assert_eq!(self.batch_size * a * b, left.len());
        assert_eq!(self.batch_size * b * c, right.len());

        let left = self.pad_matrices(&left, [a, b], [a_pad, b_pad])?;
        let right = self.pad_matrices(&right, [b, c], [b_pad, c_pad])?;

        let product = self.matmul(&left, &right, self.padded)?;

        self.pad_matrices(&product, [a_pad, c_pad], [a, c])
    }
}

impl<L, R, T> ReadValue<OpenCL, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, _offset: usize) -> Result<T, Error> {
        Err(Error::Bounds(
            "reading an individual value from a matrix multiplication is not implemented"
                .to_string(),
        ))
    }
}

pub struct Linear<T> {
    start: T,
    step: f64,
    size: usize,
    program: Program,
}

impl<T: CType> Linear<T> {
    pub fn new(start: T, step: f64, size: usize) -> Result<Self, Error> {
        programs::constructors::range(T::TYPE).map(|program| Self {
            start,
            step,
            size,
            program,
        })
    }
}

impl<T: Send + Sync> Op for Linear<T> {
    fn size(&self) -> usize {
        self.size
    }
}

impl<T: CType> Enqueue<OpenCL, T> for Linear<T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let queue = OpenCL::queue(self.size, &[])?;

        let buffer = Buffer::builder()
            .queue(queue.clone())
            .len(self.size)
            .build()?;

        let kernel = Kernel::builder()
            .name("range")
            .queue(queue)
            .program(&self.program)
            .global_work_size(self.size)
            .arg(self.start)
            .arg(self.step)
            .arg(&buffer)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(buffer)
    }
}

impl<T: CType> ReadValue<OpenCL, T> for Linear<T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        Ok(T::add(self.start, T::from_f64((offset as f64) * self.step)))
    }
}

pub struct RandomNormal {
    program: Program,
    size: usize,
}

impl RandomNormal {
    pub fn new(size: usize) -> Result<Self, Error> {
        programs::constructors::random_normal().map(|program| Self { program, size })
    }
}

impl Op for RandomNormal {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<OpenCL, f32> for RandomNormal {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let queue = OpenCL::queue(self.size, &[])?;
        let seed: u32 = Rand::new().gen();

        let buffer = Buffer::builder()
            .queue(queue.clone())
            .len(WG_SIZE * self.size().div_ceil(WG_SIZE))
            .build()?;

        let kernel = Kernel::builder()
            .name("random_normal")
            .queue(queue.clone())
            .program(&self.program)
            .global_work_size(buffer.len())
            .local_work_size(WG_SIZE)
            .arg(u64::try_from(seed).expect("seed"))
            .arg(&buffer)
            .arg_local::<f32>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? }

        if buffer.len() == self.size {
            Ok(buffer)
        } else {
            let output = Buffer::builder().queue(queue).len(self.size).build()?;

            buffer.copy(&output, Some(0), Some(self.size)).enq()?;

            Ok(output)
        }
    }
}

impl ReadValue<OpenCL, f32> for RandomNormal {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Err(Error::Bounds(
            "cannot read an individual value from a random normal distribution".to_string(),
        ))
    }
}

pub struct RandomUniform {
    program: Program,
    size: usize,
}

impl RandomUniform {
    pub fn new(size: usize) -> Result<Self, Error> {
        programs::constructors::random_uniform().map(|program| Self { program, size })
    }
}

impl Op for RandomUniform {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<OpenCL, f32> for RandomUniform {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let queue = OpenCL::queue(self.size, &[])?;
        let seed: u32 = Rand::new().gen();

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.size)
            .build()?;

        let kernel = Kernel::builder()
            .name("random_uniform")
            .queue(queue)
            .program(&self.program)
            .global_work_size(output.len())
            .arg(seed as u64)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl ReadValue<OpenCL, f32> for RandomUniform {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Ok(Rand::new().gen())
    }
}

pub struct Reduce<A, T: CType> {
    access: A,
    stride: usize,
    fold: Program,
    reduce: Program,
    reduce_all: fn(OpenCL, AccessBuf<Buffer<T>>) -> Result<T, Error>,
    id: T,
}

impl<A, T: CType> Reduce<A, T> {
    fn new(
        access: A,
        stride: usize,
        reduce: &'static str,
        reduce_all: fn(OpenCL, AccessBuf<Buffer<T>>) -> Result<T, Error>,
        id: T,
    ) -> Result<Self, Error> {
        let fold = programs::reduce::fold_axis(T::TYPE, reduce)?;
        let reduce = programs::reduce::reduce_axis(T::TYPE, reduce)?;

        Ok(Self {
            access,
            stride,
            fold,
            reduce,
            reduce_all,
            id,
        })
    }

    pub fn max(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "max",
            <OpenCL as ReduceAll<AccessBuf<Buffer<T>>, T>>::max,
            T::MIN,
        )
    }

    pub fn min(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "min",
            <OpenCL as ReduceAll<AccessBuf<Buffer<T>>, T>>::min,
            T::MAX,
        )
    }

    pub fn product(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "mul",
            <OpenCL as ReduceAll<AccessBuf<Buffer<T>>, T>>::product,
            T::ONE,
        )
    }

    pub fn sum(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "add",
            <OpenCL as ReduceAll<AccessBuf<Buffer<T>>, T>>::sum,
            T::ZERO,
        )
    }

    fn fold(
        &self,
        queue: Queue,
        input: &Buffer<T>,
        reduce_dim: usize,
        target_dim: usize,
    ) -> Result<Buffer<T>, Error> {
        let output_size = (input.len() / reduce_dim) * target_dim;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(output_size)
            .fill_val(T::ZERO)
            .build()?;

        let kernel = Kernel::builder()
            .name("fold_axis")
            .program(&self.fold)
            .queue(queue)
            .global_work_size(output_size)
            .arg(reduce_dim as u64)
            .arg(target_dim as u64)
            .arg(self.id)
            .arg(input)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn reduce(
        &self,
        queue: Queue,
        input: &Buffer<T>,
        stride: usize,
        wg_size: usize,
    ) -> Result<Buffer<T>, Error> {
        debug_assert_eq!(input.len() % stride, 0);

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len() / stride)
            .fill_val(T::ZERO)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce_axis")
            .program(&self.reduce)
            .queue(queue.clone())
            .local_work_size(wg_size)
            .global_work_size(input.len())
            .arg(self.id)
            .arg(input)
            .arg(&output)
            .arg_local::<T>(wg_size)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A: Access<T>, T: CType> Op for Reduce<A, T> {
    fn size(&self) -> usize {
        debug_assert_eq!(self.access.size() % self.stride, 0);
        self.access.size() / self.stride
    }
}

impl<A: Access<T>, T: CType> Enqueue<OpenCL, T> for Reduce<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;

        let queue = OpenCL::queue(input.len(), &[input.default_queue()])?;

        let output_size = input.len() / self.stride;

        let mut stride = self.stride;

        let log = (stride as f32).log(WG_SIZE as f32).fract();
        let target_dim = WG_SIZE.pow(log as u32);
        let mut buffer = self.fold(queue.clone(), &*input, stride, target_dim)?;

        stride = target_dim;
        debug_assert_eq!(output_size * stride, buffer.len());

        while buffer.len() > output_size {
            let wg_size = if stride < WG_SIZE {
                stride
            } else {
                debug_assert_eq!(stride % WG_SIZE, 0);
                WG_SIZE
            };

            buffer = self.reduce(queue.clone(), &buffer, stride, wg_size)?;
            stride /= wg_size;
            debug_assert_eq!(output_size * stride, buffer.len());
        }

        assert_eq!(buffer.len(), output_size);

        Ok(buffer)
    }
}

impl<A: Access<T>, T: CType> ReadValue<OpenCL, T> for Reduce<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let input = self.access.read()?.to_cl()?;
        let slice = input.create_sub_buffer(None, offset, offset + self.stride)?;
        (self.reduce_all)(OpenCL, AccessBuf::from(slice))
    }
}

pub struct Scalar<A, IT, OT> {
    access: A,
    scalar: IT,
    program: Program,
    op: fn(IT, IT) -> OT,
}

impl<A, T: CType> Scalar<A, T, T> {
    pub fn new(
        access: A,
        scalar: T,
        program: &'static str,
        op: fn(T, T) -> T,
    ) -> Result<Self, Error> {
        programs::elementwise::dual(T::TYPE, program)
            .map(|program| Self {
                access,
                scalar,
                program,
                op,
            })
            .map_err(Error::from)
    }

    pub fn add(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "add", T::add)
    }

    pub fn div(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "div", T::div)
    }

    pub fn log(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "_log", |a, e| {
            T::from_float(a.to_float().log(e.to_float()))
        })
    }

    pub fn mul(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "mul", T::mul)
    }

    pub fn pow(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "pow", T::pow)
    }

    pub fn rem(access: A, scalar: T) -> Result<Self, Error> {
        let program = if T::IS_FLOAT { "fmod" } else { "mod" };
        Self::new(access, scalar, program, T::rem)
    }

    pub fn sub(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "sub", T::sub)
    }
}

impl<A, T> Scalar<A, T, u8>
where
    T: CType,
{
    fn compare(
        access: A,
        scalar: T,
        program: &'static str,
        op: fn(T, T) -> u8,
    ) -> Result<Self, Error> {
        programs::elementwise::dual_boolean(T::TYPE, program)
            .map(|program| Self {
                access,
                scalar,
                program,
                op,
            })
            .map_err(Error::from)
    }

    pub fn and(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "and", |l, r| {
            if l != T::ZERO && r != T::ZERO {
                1
            } else {
                0
            }
        })
    }

    pub fn or(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "or", |l, r| {
            if l != T::ZERO || r != T::ZERO {
                1
            } else {
                0
            }
        })
    }

    pub fn xor(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "xor", |l, r| {
            if (l != T::ZERO) ^ (r != T::ZERO) {
                1
            } else {
                0
            }
        })
    }

    pub fn eq(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "eq", |l, r| if l == r { 1 } else { 0 })
    }

    pub fn ge(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "ge", |l, r| if l >= r { 1 } else { 0 })
    }

    pub fn gt(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "gt", |l, r| if l > r { 1 } else { 0 })
    }

    pub fn le(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "le", |l, r| if l <= r { 1 } else { 0 })
    }

    pub fn lt(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "lt", |l, r| if l < r { 1 } else { 0 })
    }

    pub fn ne(access: A, scalar: T) -> Result<Self, Error> {
        Self::compare(access, scalar, "ne", |l, r| if l != r { 1 } else { 0 })
    }
}

impl<A, IT, OT> Op for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, IT, OT> Enqueue<OpenCL, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;

        debug_assert_eq!(input.len(), self.size());

        let queue = OpenCL::queue(input.len(), &[input.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("dual_scalar")
            .program(&self.program)
            .queue(queue)
            .global_work_size(input.len())
            .arg(&*input)
            .arg(self.scalar)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A, IT, OT> ReadValue<OpenCL, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access
            .read_value(offset)
            .map(|n| (self.op)(n, self.scalar))
    }
}

pub struct Slice<A, T> {
    access: A,
    spec: SliceSpec,
    read: Program,
    write: Option<Program>,
    write_value: Option<Program>,
    dtype: PhantomData<T>,
}

impl<A, T: CType> Slice<A, T> {
    pub fn new(access: A, shape: &[usize], range: Range) -> Result<Self, Error> {
        let spec = SliceSpec::new(shape, range);

        let read = programs::slice::read_slice(T::TYPE, spec.clone())?;

        Ok(Self {
            access,
            spec,
            read,
            write: None,
            write_value: None,
            dtype: PhantomData,
        })
    }
}

impl<A: Send + Sync, T: Send + Sync> Op for Slice<A, T> {
    fn size(&self) -> usize {
        self.spec.size()
    }
}

impl<A: Access<T>, T: CType> Enqueue<OpenCL, T> for Slice<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read()?.to_cl()?;
        let queue = OpenCL::queue(self.size(), &[source.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.size())
            .build()?;

        let kernel = Kernel::builder()
            .name("read_slice")
            .program(&self.read)
            .queue(queue)
            .global_work_size(output.len())
            .arg(&*source)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A: Access<T>, T: CType> ReadValue<OpenCL, T> for Slice<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.access.read_value(self.spec.source_offset(offset))
    }
}

impl<A, T> Write<OpenCL, T> for Slice<A, T>
where
    T: CType,
    A: AccessMut<T> + fmt::Debug,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        let data = data.to_cl()?;
        let size_hint = self.size();
        let source = self.access.cl_buffer()?;

        let queue = OpenCL::queue(size_hint, &[source.default_queue()])?;

        if self.write.is_none() {
            let program = programs::slice::write_to_slice(T::TYPE, self.spec.clone())?;
            self.write = Some(program);
        }

        let kernel = Kernel::builder()
            .name("write_slice")
            .program(self.write.as_ref().expect("CL write op"))
            .queue(queue)
            .global_work_size(source.len())
            .arg(source)
            .arg(&*data)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(())
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        let size_hint = self.size();
        let source = self.access.cl_buffer()?;

        let queue = OpenCL::queue(size_hint, &[source.default_queue()])?;

        if self.write.is_none() {
            let program = programs::slice::write_value_to_slice(T::TYPE, self.spec.clone())?;
            self.write_value = Some(program);
        }

        let kernel = Kernel::builder()
            .name("write_slice_value")
            .program(self.write_value.as_ref().expect("CL write op"))
            .queue(queue)
            .global_work_size(source.len())
            .arg(source)
            .arg(value)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(())
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.access
            .borrow_mut()
            .write_value_at(self.spec.source_offset(offset), value)
    }
}

pub struct Unary<A, IT, OT> {
    access: A,
    program: Program,
    op: fn(IT) -> OT,
    dtype: PhantomData<(IT, OT)>,
}

impl<A, IT: CType, OT: CType> Unary<A, IT, OT> {
    fn new(access: A, program: &'static str, op: fn(IT) -> OT) -> Result<Self, Error> {
        let program = programs::elementwise::unary(IT::Float::TYPE, IT::TYPE, OT::TYPE, program)?;

        Ok(Self {
            access,
            program,
            op,
            dtype: PhantomData,
        })
    }
}

impl<A, T: CType> Unary<A, T, T> {
    pub fn abs(access: A) -> Result<Self, Error> {
        Self::new(access, "abs", |n| T::from_float(n.to_float().ln()))
    }

    pub fn exp(access: A) -> Result<Self, Error> {
        Self::new(access, "exp", |n| T::from_float(n.to_float().ln()))
    }

    pub fn ln(access: A) -> Result<Self, Error> {
        Self::new(access, "_log", |n| T::from_float(n.to_float().ln()))
    }

    pub fn round(access: A) -> Result<Self, Error> {
        Self::new(access, "round", |n| T::from_float(n.to_float().ln()))
    }
}

impl<A, T: CType> Unary<A, T, T::Float> {
    pub fn sin(access: A) -> Result<Self, Error> {
        Self::new(access, "sin", |n| n.to_float().sin())
    }

    pub fn sinh(access: A) -> Result<Self, Error> {
        Self::new(access, "sinh", |n| n.to_float().sinh())
    }

    pub fn asin(access: A) -> Result<Self, Error> {
        Self::new(access, "asin", |n| n.to_float().asin())
    }

    pub fn cos(access: A) -> Result<Self, Error> {
        Self::new(access, "cos", |n| n.to_float().cos())
    }

    pub fn cosh(access: A) -> Result<Self, Error> {
        Self::new(access, "cosh", |n| n.to_float().cosh())
    }

    pub fn acos(access: A) -> Result<Self, Error> {
        Self::new(access, "acos", |n| n.to_float().acos())
    }
    pub fn tan(access: A) -> Result<Self, Error> {
        Self::new(access, "tan", |n| n.to_float().tan())
    }

    pub fn tanh(access: A) -> Result<Self, Error> {
        Self::new(access, "tanh", |n| n.to_float().tanh())
    }

    pub fn atan(access: A) -> Result<Self, Error> {
        Self::new(access, "atan", |n| n.to_float().atan())
    }
}

impl<A, T: CType> Unary<A, T, u8> {
    pub fn not(access: A) -> Result<Self, Error> {
        Self::new(access, "not", |n| if n == T::ZERO { 1 } else { 0 })
    }
}

impl<A, T: Float> Unary<A, T, u8> {
    pub fn inf(access: A) -> Result<Self, Error> {
        Self::new(access, "isinf", |n| if n.is_inf() { 1 } else { 0 })
    }

    pub fn nan(access: A) -> Result<Self, Error> {
        Self::new(access, "isnan", |n| if n.is_nan() { 1 } else { 0 })
    }
}

impl<A, IT, OT> Op for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, IT, OT> Enqueue<OpenCL, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;
        let queue = OpenCL::queue(input.len(), &[input.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("unary")
            .program(&self.program)
            .queue(queue)
            .global_work_size(input.len())
            .arg(&*input)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A: Access<IT>, IT: CType, OT: CType> ReadValue<OpenCL, OT> for Unary<A, IT, OT> {
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access.read_value(offset).map(|n| (self.op)(n))
    }
}

pub struct View<A, T> {
    access: A,
    program: Program,
    size: usize,
    spec: ViewSpec,
    dtype: PhantomData<T>,
}

impl<A, T> View<A, T>
where
    T: CType,
{
    fn new(access: A, spec: ViewSpec) -> Result<Self, Error> {
        let size = spec.shape.iter().product();
        let program = programs::view::view(T::TYPE, spec.clone())?;

        Ok(Self {
            access,
            program,
            size,
            spec,
            dtype: PhantomData,
        })
    }

    pub fn broadcast(access: A, shape: Shape, broadcast: Shape) -> Result<Self, Error> {
        let strides = strides_for(&shape, shape.len()).collect();
        let spec = ViewSpec::new(broadcast, strides);
        Self::new(access, spec)
    }

    pub fn transpose(access: A, shape: Shape, axes: Axes) -> Result<Self, Error> {
        let strides = strides_for(&shape, shape.len()).collect::<Strides>();
        let shape = axes.iter().copied().map(|x| shape[x]).collect::<Shape>();
        let strides = axes.into_iter().map(|x| strides[x]).collect::<Strides>();
        let spec = ViewSpec::new(shape, strides);
        Self::new(access, spec)
    }
}

impl<A, T> Op for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.size
    }
}

impl<A: Access<T>, T: CType> Enqueue<OpenCL, T> for View<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read()?.to_cl()?;

        let queue = OpenCL::queue(self.size, &[source.default_queue()])?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.size)
            .build()?;

        let kernel = Kernel::builder()
            .name("view")
            .program(&self.program)
            .queue(queue)
            .global_work_size(self.size)
            .arg(&*source)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A: Access<T>, T: CType> ReadValue<OpenCL, T> for View<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.access.read_value(self.spec.source_offset(offset))
    }
}

#[inline]
fn pad_dim(dim: usize, size: usize) -> usize {
    size * dim.div_ceil(size)
}

#[allow(unused)]
fn inspect<T: CType>(name: &'static str, buffer: &Buffer<T>) -> Result<(), Error> {
    let mut inspect = vec![T::ZERO; buffer.len()];
    buffer.read(inspect.as_mut_slice()).enq()?;
    Ok(())
}
