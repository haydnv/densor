use std::f32::consts::PI;
use std::iter;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

use rand::Rng;
use rayon::join;
use rayon::prelude::*;

use crate::access::{Access, AccessBuffer};
use crate::buffer::BufferConverter;
use crate::ops::Op;
use crate::{
    stackvec, strides_for, AxisRange, CType, Enqueue, Error, Float, Range, Shape, Strides,
};

use super::buffer::Buffer;
use super::platform::{Heap, Host, Stack};
use super::{SliceConverter, StackVec, VEC_MIN_SIZE};

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    cmp: fn(T, T) -> u8,
}

impl<L, R, T> Op for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T: CType> Compare<L, R, T> {
    pub fn eq(left: L, right: R) -> Self {
        Self {
            left,
            right,
            cmp: |l, r| if l == r { 1 } else { 0 },
        }
    }
}

impl<L, R, T> Enqueue<Stack> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = StackVec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual(self.cmp, left, right)
    }
}

impl<L, R, T> Enqueue<Heap> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Vec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual_parallel(self.cmp, left, right)
    }
}

impl<L, R, T> Enqueue<Host> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
    }
}

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    zip: fn(T, T) -> T,
}

impl<L, R, T> Op for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T: CType> Dual<L, R, T> {
    pub fn add(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: Add::add,
        }
    }

    pub fn sub(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: Sub::sub,
        }
    }
}

impl<L, R, T> Enqueue<Stack> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual(self.zip, left, right)
    }
}

impl<L, R, T> Enqueue<Heap> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual_parallel(self.zip, left, right)
    }
}

impl<L, R, T> Enqueue<Host> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::from)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::from)
        }
    }
}

pub struct Linear<T> {
    start: T,
    step: f64,
    size: usize,
}

impl<T> Linear<T> {
    pub fn new(start: T, step: f64, size: usize) -> Self {
        Self { start, step, size }
    }
}

impl<T: Send + Sync> Op for Linear<T> {
    fn size(&self) -> usize {
        self.size
    }
}

impl<T: CType> Enqueue<Stack> for Linear<T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let start = self.start.to_float().to_f64();

        let buffer = (0..self.size)
            .into_iter()
            .map(|i| i as f64)
            .map(|i| i * self.step)
            .map(|o| start + o)
            .map(T::from_f64)
            .collect();

        Ok(buffer)
    }
}

impl<T: CType> Enqueue<Heap> for Linear<T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let start = self.start.to_float().to_f64();

        let buffer = (0..self.size)
            .into_par_iter()
            .map(|i| i as f64)
            .map(|i| i * self.step)
            .map(|o| start + o)
            .map(T::from_f64)
            .collect();

        Ok(buffer)
    }
}

impl<T: CType> Enqueue<Host> for Linear<T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
    }
}

pub struct RandomNormal {
    size: usize,
}

impl RandomNormal {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    fn box_muller(u: [f32; 2]) -> [f32; 2] {
        let [u1, u2] = u;
        let r = (u1.ln() * -2.).sqrt();
        let theta = 2. * PI * u2;
        [r * theta.cos(), r * theta.sin()]
    }
}

impl Op for RandomNormal {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<Heap> for RandomNormal {
    type Buffer = Vec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut u = vec![
            0.0f32;
            if self.size % 2 == 0 {
                self.size
            } else {
                self.size + 1
            }
        ];

        rand::thread_rng().fill(&mut u[..]);

        let mut output = u
            .par_chunks_exact(2)
            .map(|u| {
                let u: [f32; 2] = u.try_into().expect("u");
                Self::box_muller(u)
            })
            .flatten()
            .collect::<Vec<f32>>();

        if output.len() > self.size {
            output.pop();
        }

        debug_assert_eq!(output.len(), self.size);

        Ok(output)
    }
}

impl Enqueue<Stack> for RandomNormal {
    type Buffer = StackVec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut rng = rand::thread_rng();

        let mut output = iter::repeat_with(|| [rng.gen(), rng.gen()])
            .take(self.size.div_ceil(2))
            .map(Self::box_muller)
            .flatten()
            .collect::<StackVec<f32>>();

        if output.len() > self.size {
            output.pop();
        }

        debug_assert_eq!(output.len(), self.size);

        Ok(output)
    }
}

impl Enqueue<Host> for RandomNormal {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
    }
}

pub struct RandomUniform {
    size: usize,
}

impl RandomUniform {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Op for RandomUniform {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<Heap> for RandomUniform {
    type Buffer = Vec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut data = vec![0.; self.size];
        rand::thread_rng().fill(&mut data[..]);
        Ok(data)
    }
}

impl Enqueue<Stack> for RandomUniform {
    type Buffer = StackVec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut data = stackvec![0.; self.size];
        rand::thread_rng().fill(&mut data[..]);
        Ok(data)
    }
}

impl Enqueue<Host> for RandomUniform {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
    }
}

pub struct Slice<A, T> {
    access: A,
    source_strides: Strides,
    range: Range,
    shape: Shape,
    strides: Strides,
    dtype: PhantomData<T>,
}

impl<A, T> Slice<A, T> {
    pub fn new(access: A, shape: &[usize], range: Range) -> Self {
        let source_strides = strides_for(shape, shape.len());
        let shape = range.iter().filter_map(|ar| ar.size()).collect::<Shape>();
        let strides = strides_for(&shape, shape.len());

        Self {
            access,
            source_strides,
            range,
            shape,
            strides,
            dtype: PhantomData,
        }
    }
}

impl<A: Send + Sync, T: Copy + Send + Sync> Slice<A, T> {
    fn source_offset(&self, offset: usize) -> usize {
        debug_assert!(!self.shape.is_empty());
        debug_assert_eq!(self.shape.len(), self.strides.len());

        let mut coord = self
            .strides
            .iter()
            .copied()
            .zip(&self.shape)
            .map(|(stride, dim)| {
                if stride == 0 {
                    0
                } else {
                    (offset / stride) % dim
                }
            });

        let mut offset = 0;
        for (stride, bound) in self.source_strides.iter().zip(self.range.iter()) {
            let i = match bound {
                AxisRange::At(i) => *i,
                AxisRange::In(start, stop, step) => {
                    let i = start + (coord.next().expect("i") * step);
                    debug_assert!(i < *stop);
                    i
                }
                AxisRange::Of(indices) => indices[coord.next().expect("i")],
            };

            offset += i * stride;
        }

        offset
    }

    fn read(&self, source: &[T]) -> Result<StackVec<T>, Error> {
        let output = (0..self.size())
            .into_iter()
            .map(|offset_out| self.source_offset(offset_out))
            .map(|offset_in| source[offset_in])
            .collect();

        Ok(output)
    }

    fn read_parallel(&self, source: &[T]) -> Result<Vec<T>, Error> {
        let output = (0..self.size())
            .into_par_iter()
            .map(|offset_out| self.source_offset(offset_out))
            .map(|offset_in| source[offset_in])
            .collect();

        Ok(output)
    }
}

impl<B, T> Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    fn overwrite(&mut self, data: &[T]) -> Result<(), Error> {
        if data.len() == self.size() {
            for (offset, value) in data.into_iter().copied().enumerate() {
                let source_offset = self.source_offset(offset);
                let source = self.access.as_mut().into_inner();
                source.as_mut()[source_offset] = value;
            }

            Ok(())
        } else {
            Err(Error::Bounds(format!(
                "cannot overwrite a slice of size {} with a buffer of size {}",
                self.size(),
                data.len(),
            )))
        }
    }
}

impl<A: Send + Sync, T: Send + Sync> Op for Slice<A, T> {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<A: Access<T>, T: CType> Enqueue<Heap> for Slice<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .and_then(|buf| self.read_parallel(&*buf))
    }
}

impl<A: Access<T>, T: CType> Enqueue<Stack> for Slice<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .and_then(|buf| self.read(&*buf))
    }
}

impl<A: Access<T>, T: CType> Enqueue<Host> for Slice<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<'a, B, T> crate::ops::Write<'a, Heap> for Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    type Data = &'a [T];

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.overwrite(data)
    }
}

impl<'a, B, T> crate::ops::Write<'a, Stack> for Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    type Data = &'a [T];

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.overwrite(data)
    }
}

impl<'a, B, T> crate::ops::Write<'a, Host> for Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    type Data = SliceConverter<'a, T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.overwrite(&*data)
    }
}

pub struct Unary<A, IT, OT> {
    access: A,
    op: fn(IT) -> OT,
}

impl<A, T> Unary<A, T, T>
where
    A: Access<T>,
    T: CType,
{
    pub fn ln(access: A) -> Self {
        Self {
            access,
            op: |n| T::from_float(n.to_float().ln()),
        }
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

impl<A, IT, OT> Enqueue<Heap> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Vec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|input| input.into_par_iter().copied().map(self.op).collect())
    }
}

impl<A, IT, OT> Enqueue<Stack> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = StackVec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|input| input.into_iter().copied().map(self.op).collect())
    }
}

impl<A, IT, OT> Enqueue<Host> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
    }
}

struct ViewSpec<T> {
    shape: Shape,
    strides: Strides,
    source_strides: Strides,
    dtype: PhantomData<T>,
}

impl<T: CType> ViewSpec<T> {
    fn invert_offset(&self, offset: usize) -> usize {
        debug_assert!(offset < self.shape.iter().product::<usize>());

        self.strides
            .iter()
            .copied()
            .zip(self.shape.iter().copied())
            .map(|(stride, dim)| {
                if stride == 0 {
                    0
                } else {
                    (offset / stride) % dim
                }
            }) // coord
            .zip(self.source_strides.iter().copied())
            .map(|(i, source_stride)| i * source_stride) // source offset
            .sum::<usize>()
    }

    fn read(&self, source: BufferConverter<T>) -> Result<StackVec<T>, Error> {
        let source = source.to_slice()?;

        let buffer = (0..self.shape.iter().product())
            .into_iter()
            .map(|offset| self.invert_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }

    fn read_parallel(&self, source: BufferConverter<T>) -> Result<Vec<T>, Error> {
        let source = source.to_slice()?;

        let buffer = (0..self.shape.iter().product())
            .into_par_iter()
            .map(|offset| self.invert_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }
}

pub struct View<A, T> {
    access: A,
    spec: ViewSpec<T>,
}

impl<A, T> View<A, T>
where
    A: Access<T>,
    T: CType,
{
    pub fn new(access: A, shape: Shape, broadcast: Shape) -> Self {
        let strides = strides_for(&shape, broadcast.len());
        let source_strides = strides_for(&shape, shape.len());

        Self {
            access,
            spec: ViewSpec {
                shape: broadcast,
                strides,
                source_strides,
                dtype: PhantomData,
            },
        }
    }
}

impl<A, T> Op for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.spec.shape.iter().product()
    }
}

impl<A, T> Enqueue<Stack> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access.read().and_then(|source| self.spec.read(source))
    }
}

impl<A, T> Enqueue<Heap> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|source| self.spec.read_parallel(source))
    }
}

impl<A, T> Enqueue<Host> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
    }
}

fn exec_dual<IT, OT>(
    zip: fn(IT, IT) -> OT,
    left: BufferConverter<IT>,
    right: BufferConverter<IT>,
) -> Result<StackVec<OT>, Error>
where
    IT: CType,
    OT: CType,
{
    let left = left.to_slice()?;
    let right = right.to_slice()?;

    let output = left
        .into_iter()
        .copied()
        .zip(right.into_iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

fn exec_dual_parallel<IT, OT>(
    zip: fn(IT, IT) -> OT,
    left: BufferConverter<IT>,
    right: BufferConverter<IT>,
) -> Result<Vec<OT>, Error>
where
    IT: CType,
    OT: CType,
{
    let left = left.to_slice()?;
    let right = right.to_slice()?;

    let output = left
        .into_par_iter()
        .copied()
        .zip(right.into_par_iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

#[inline]
fn try_join<'a, L, R, T>(
    left: &'a L,
    right: &'a R,
) -> Result<(BufferConverter<'a, T>, BufferConverter<'a, T>), Error>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    let (l, r) = join(|| left.read(), || right.read());

    Ok((l?, r?))
}
