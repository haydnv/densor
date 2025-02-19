use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::marker::PhantomData;

use crate::access::*;
use crate::buffer::BufferInstance;
use crate::ops::*;
use crate::platform::PlatformInstance;
use crate::{
    range_shape, shape, strides_for, Axes, AxisRange, BufferConverter, CType, Constant, Convert,
    Error, Float, Platform, Range, Shape,
};

pub struct Array<T, A, P> {
    shape: Shape,
    access: A,
    platform: P,
    dtype: PhantomData<T>,
}

impl<T, A: Clone, P: Clone> Clone for Array<T, A, P> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            access: self.access.clone(),
            platform: self.platform.clone(),
            dtype: self.dtype,
        }
    }
}

impl<T, A, P> Array<T, A, P> {
    fn apply<O, OT, Op>(self, op: Op) -> Result<Array<OT, AccessOp<O, P>, P>, Error>
    where
        P: Copy,
        Op: Fn(P, A) -> Result<AccessOp<O, P>, Error>,
    {
        let access = (op)(self.platform, self.access)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform: self.platform,
            dtype: PhantomData,
        })
    }

    fn reduce_axes<Op>(
        self,
        mut axes: Axes,
        keepdims: bool,
        op: Op,
    ) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error>
    where
        T: CType,
        A: Access<T>,
        P: Transform<A, T> + ReduceAxes<Accessor<T>, T>,
        Op: Fn(P, Accessor<T>, usize) -> Result<AccessOp<P::Op, P>, Error>,
        Accessor<T>: From<A> + From<AccessOp<P::Transpose, P>>,
    {
        axes.sort();
        axes.dedup();

        let shape = reduce_axes(&self.shape, &axes, keepdims)?;
        let size = shape.iter().product::<usize>();
        let stride = axes.iter().copied().map(|x| self.shape[x]).product();
        let platform = P::select(size);

        let access = permute_for_reduce(self.platform, self.access, self.shape, axes)?;
        let access = (op)(self.platform, access, stride)?;

        Ok(Array {
            access,
            shape,
            platform,
            dtype: PhantomData,
        })
    }

    pub fn access(&self) -> &A {
        &self.access
    }

    pub fn into_access(self) -> A {
        self.access
    }
}

impl<T, L, P> Array<T, L, P> {
    fn apply_dual<O, OT, R, Op>(
        self,
        other: Array<T, R, P>,
        op: Op,
    ) -> Result<Array<OT, AccessOp<O, P>, P>, Error>
    where
        P: Copy,
        Op: Fn(P, L, R) -> Result<AccessOp<O, P>, Error>,
    {
        let access = (op)(self.platform, self.access, other.access)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

// constructors
impl<T: CType> Array<T, Accessor<T>, Platform> {
    pub fn from<A, P>(array: Array<T, A, P>) -> Self
    where
        Accessor<T>: From<A>,
        Platform: From<P>,
    {
        Self {
            shape: array.shape,
            access: array.access.into(),
            platform: array.platform.into(),
            dtype: array.dtype,
        }
    }
}

impl<T, B, P> Array<T, AccessBuf<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    fn new_inner(platform: P, buffer: B, shape: Shape) -> Result<Self, Error> {
        if !shape.is_empty() && shape.iter().product::<usize>() == buffer.len() {
            let access = buffer.into();

            Ok(Self {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot construct an array with shape {shape:?} from a buffer of size {}",
                buffer.len(),
            )))
        }
    }

    pub fn convert<'a, FB>(buffer: FB, shape: Shape) -> Result<Self, Error>
    where
        FB: Into<BufferConverter<'a, T>>,
        P: Convert<T, Buffer = B>,
    {
        let buffer = buffer.into();
        let platform = P::select(buffer.len());
        let buffer = platform.convert(buffer)?;
        Self::new_inner(platform, buffer, shape)
    }

    pub fn new(buffer: B, shape: Shape) -> Result<Self, Error> {
        let platform = P::select(buffer.len());
        Self::new_inner(platform, buffer, shape)
    }
}

impl<T, P> Array<T, AccessBuf<P::Buffer>, P>
where
    T: CType,
    P: Constant<T>,
{
    pub fn constant(value: T, shape: Shape) -> Result<Self, Error> {
        if !shape.is_empty() {
            let size = shape.iter().product();
            let platform = P::select(size);
            let buffer = platform.constant(value, size)?;
            let access = buffer.into();

            Ok(Self {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::Bounds(
                "cannot construct an array with an empty shape".to_string(),
            ))
        }
    }
}

impl<T, P> Array<T, AccessBuf<P::Buffer>, P>
where
    T: CType,
    P: Convert<T>,
{
    pub fn copy<A: Access<T>>(source: &Array<T, A, P>) -> Result<Self, Error> {
        let buffer = source
            .buffer()
            .and_then(|buf| source.platform.convert(buf))?;

        Ok(Self {
            shape: source.shape.clone(),
            access: buffer.into(),
            platform: source.platform,
            dtype: source.dtype,
        })
    }
}

// op constructors
impl<T: CType, P: PlatformInstance> Array<T, AccessOp<P::Range, P>, P>
where
    P: Construct<T>,
{
    pub fn range(start: T, stop: T, shape: Shape) -> Result<Self, Error> {
        let size = shape.iter().product();
        let platform = P::select(size);

        platform.range(start, stop, size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Normal, P>, P>
where
    P: Random,
{
    pub fn random_normal(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_normal(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Uniform, P>, P>
where
    P: Random,
{
    pub fn random_uniform(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_uniform(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

// references
impl<T, B, P> Array<T, AccessBuf<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    pub fn as_mut<RB: ?Sized>(&mut self) -> Array<T, AccessBuf<&mut RB>, P>
    where
        B: BorrowMut<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_mut(),
            platform: self.platform,
            dtype: PhantomData,
        }
    }

    pub fn as_ref<RB: ?Sized>(&self) -> Array<T, AccessBuf<&RB>, P>
    where
        B: Borrow<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_ref(),
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

impl<T, O, P> Array<T, AccessOp<O, P>, P>
where
    T: CType,
    O: Enqueue<P, T>,
    P: PlatformInstance,
{
    pub fn as_mut<'a>(&'a mut self) -> Array<T, &'a mut AccessOp<O, P>, P>
    where
        O: Write<P, T>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: &mut self.access,
            platform: self.platform,
            dtype: PhantomData,
        }
    }

    pub fn as_ref(&self) -> Array<T, &AccessOp<O, P>, P> {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: &self.access,
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

// traits

/// An n-dimensional array
pub trait NDArray: Send + Sync {
    /// The data type of the elements in this array
    type DType: CType;

    /// The platform used to construct operations on this array.
    type Platform: PlatformInstance;

    /// Return the number of dimensions in this array.
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Return the number of elements in this array.
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Borrow the shape of this array.
    fn shape(&self) -> &[usize];
}

impl<T, A, P> NDArray for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: PlatformInstance,
{
    type DType = T;
    type Platform = P;

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Access methods for an [`NDArray`]
pub trait NDArrayRead: NDArray + fmt::Debug + Sized {
    /// Read the value of this [`NDArray`] into a [`BufferConverter`].
    fn buffer(&self) -> Result<BufferConverter<Self::DType>, Error>;

    /// Buffer this [`NDArray`] into a new, owned array, allocating only if needed.
    fn into_read(
        self,
    ) -> Result<
        Array<
            Self::DType,
            AccessBuf<<Self::Platform as Convert<Self::DType>>::Buffer>,
            Self::Platform,
        >,
        Error,
    >
    where
        Self::Platform: Convert<Self::DType>;

    /// Read the value at a specific `coord` in this [`NDArray`].
    fn read_value(&self, coord: &[usize]) -> Result<Self::DType, Error>;
}

impl<T, A, P> NDArrayRead for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: PlatformInstance,
{
    fn buffer(&self) -> Result<BufferConverter<T>, Error> {
        self.access.read()
    }

    fn into_read(self) -> Result<Array<Self::DType, AccessBuf<P::Buffer>, Self::Platform>, Error>
    where
        P: Convert<T>,
    {
        let buffer = self.buffer().and_then(|buf| self.platform.convert(buf))?;
        debug_assert_eq!(buffer.len(), self.size());

        Ok(Array {
            shape: self.shape,
            access: buffer.into(),
            platform: self.platform,
            dtype: self.dtype,
        })
    }

    fn read_value(&self, coord: &[usize]) -> Result<T, Error> {
        valid_coord(coord, self.shape())?;

        let strides = strides_for(self.shape(), self.ndim());

        let offset = coord
            .iter()
            .zip(strides)
            .map(|(i, stride)| i * stride)
            .sum();

        self.access.read_value(offset)
    }
}

/// Access methods for a mutable [`NDArray`]
pub trait NDArrayWrite: NDArray + fmt::Debug + Sized {
    /// Overwrite this [`NDArray`] with the value of the `other` array.
    fn write<O: NDArrayRead<DType = Self::DType>>(&mut self, other: &O) -> Result<(), Error>;

    /// Overwrite this [`NDArray`] with a constant scalar `value`.
    fn write_value(&mut self, value: Self::DType) -> Result<(), Error>;

    /// Write the given `value` at the given `coord` of this [`NDArray`].
    fn write_value_at(&mut self, coord: &[usize], value: Self::DType) -> Result<(), Error>;
}

// write ops
impl<T, A, P> NDArrayWrite for Array<T, A, P>
where
    T: CType,
    A: AccessMut<T>,
    P: PlatformInstance,
{
    fn write<O>(&mut self, other: &O) -> Result<(), Error>
    where
        O: NDArrayRead<DType = Self::DType>,
    {
        same_shape("write", self.shape(), other.shape())?;
        other.buffer().and_then(|buf| self.access.write(buf))
    }

    fn write_value(&mut self, value: Self::DType) -> Result<(), Error> {
        self.access.write_value(value)
    }

    fn write_value_at(&mut self, coord: &[usize], value: Self::DType) -> Result<(), Error> {
        valid_coord(coord, self.shape())?;

        let offset = coord
            .iter()
            .zip(strides_for(self.shape(), self.ndim()))
            .map(|(i, stride)| i * stride)
            .sum();

        self.access.write_value_at(offset, value)
    }
}

// op traits

/// Array cast operations
pub trait NDArrayCast<OT: CType>: NDArray + Sized {
    type Output: Access<OT>;

    /// Construct a new array cast operation.
    fn cast(self) -> Result<Array<OT, Self::Output, Self::Platform>, Error>;
}

impl<IT, OT, A, P> NDArrayCast<OT> for Array<IT, A, P>
where
    IT: CType,
    OT: CType,
    A: Access<IT>,
    P: ElementwiseCast<A, IT, OT>,
{
    type Output = AccessOp<P::Op, P>;

    fn cast(self) -> Result<Array<OT, AccessOp<P::Op, P>, P>, Error> {
        Ok(Array {
            shape: self.shape,
            access: self.platform.cast(self.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

/// Axis-wise array reduce operations
pub trait NDArrayReduce: NDArray + fmt::Debug {
    type Output: Access<Self::DType>;

    /// Construct a max-reduce operation over the given `axes`.
    fn max(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a min-reduce operation over the given `axes`.
    fn min(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a product-reduce operation over the given `axes`.
    fn product(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a sum-reduce operation over the given `axes`.
    fn sum(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayReduce for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: Transform<A, T> + ReduceAxes<Accessor<T>, T>,
    Accessor<T>: From<A> + From<AccessOp<P::Transpose, P>>,
{
    type Output = AccessOp<P::Op, P>;

    fn max(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::max(platform, access, stride)
        })
    }

    fn min(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::min(platform, access, stride)
        })
    }

    fn product(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::product(platform, access, stride)
        })
    }

    fn sum(
        self,
        axes: Axes,
        keepdims: bool,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.reduce_axes(axes, keepdims, |platform, access, stride| {
            ReduceAxes::sum(platform, access, stride)
        })
    }
}

/// Array transform operations
pub trait NDArrayTransform: NDArray + Sized + fmt::Debug {
    /// The type returned by `broadcast`
    type Broadcast: Access<Self::DType>;

    /// The type returned by `slice`
    type Slice: Access<Self::DType>;

    /// The type returned by `transpose`
    type Transpose: Access<Self::DType>;

    /// Broadcast this array into the given `shape`.
    fn broadcast(
        self,
        shape: Shape,
    ) -> Result<Array<Self::DType, Self::Broadcast, Self::Platform>, Error>;

    /// Reshape this `array`.
    fn reshape(self, shape: Shape) -> Result<Self, Error>;

    /// Construct a slice of this array.
    fn slice(self, range: Range) -> Result<Array<Self::DType, Self::Slice, Self::Platform>, Error>;

    /// Contract the given `axes` of this array.
    /// This will return an error if any of the `axes` have dimension > 1.
    fn squeeze(self, axes: Axes) -> Result<Self, Error>;

    /// Expand the given `axes` of this array.
    fn unsqueeze(self, axes: Axes) -> Result<Self, Error>;

    /// Transpose this array according to the given `permutation`.
    /// If no permutation is given, the array axes will be reversed.
    fn transpose(
        self,
        permutation: Option<Axes>,
    ) -> Result<Array<Self::DType, Self::Transpose, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayTransform for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: Transform<A, T>,
{
    type Broadcast = AccessOp<P::Broadcast, P>;
    type Slice = AccessOp<P::Slice, P>;
    type Transpose = AccessOp<P::Transpose, P>;

    fn broadcast(self, shape: Shape) -> Result<Array<T, AccessOp<P::Broadcast, P>, P>, Error> {
        if !can_broadcast(self.shape(), &shape) {
            return Err(Error::Bounds(format!(
                "cannot broadcast {self:?} into {shape:?}"
            )));
        }

        let platform = P::select(shape.iter().product());
        let broadcast = Shape::from_slice(&shape);
        let access = platform.broadcast(self.access, self.shape, broadcast)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }

    fn reshape(mut self, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == self.size() {
            self.shape = shape;
            Ok(self)
        } else {
            Err(Error::Bounds(format!(
                "cannot reshape an array with shape {:?} into {shape:?}",
                self.shape
            )))
        }
    }

    fn slice(self, mut range: Range) -> Result<Array<T, AccessOp<P::Slice, P>, P>, Error> {
        for (dim, range) in self.shape.iter().zip(&range) {
            match range {
                AxisRange::At(i) if i < dim => Ok(()),
                AxisRange::In(start, stop, _step) if start < dim && stop <= dim => Ok(()),
                AxisRange::Of(indices) if indices.iter().all(|i| i < dim) => Ok(()),
                range => Err(Error::Bounds(format!(
                    "invalid range {range:?} for dimension {dim}"
                ))),
            }?;
        }

        for dim in self.shape.iter().skip(range.len()).copied() {
            range.push(AxisRange::In(0, dim, 1));
        }

        let shape = range_shape(self.shape(), &range);
        let access = self.platform.slice(self.access, &self.shape, range)?;
        let platform = P::select(shape.iter().product());

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }

    fn squeeze(mut self, mut axes: Axes) -> Result<Self, Error> {
        if axes.iter().copied().any(|x| x >= self.ndim()) {
            return Err(Error::Bounds(format!("invalid contraction axes: {axes:?}")));
        }

        axes.sort();

        for x in axes.into_iter().rev() {
            self.shape.remove(x);
        }

        Ok(self)
    }

    fn unsqueeze(mut self, mut axes: Axes) -> Result<Self, Error> {
        if axes.iter().copied().any(|x| x > self.ndim()) {
            return Err(Error::Bounds(format!("invalid expansion axes: {axes:?}")));
        }

        axes.sort();

        for x in axes.into_iter().rev() {
            self.shape.insert(x, 1);
        }

        Ok(self)
    }

    fn transpose(
        self,
        permutation: Option<Axes>,
    ) -> Result<Array<T, AccessOp<P::Transpose, P>, P>, Error> {
        let permutation = if let Some(axes) = permutation {
            if axes.len() == self.ndim()
                && axes.iter().copied().all(|x| x < self.ndim())
                && !(1..axes.len())
                    .into_iter()
                    .any(|i| axes[i..].contains(&axes[i - 1]))
            {
                Ok(axes)
            } else {
                Err(Error::Bounds(format!(
                    "invalid permutation for shape {:?}: {:?}",
                    self.shape, axes
                )))
            }
        } else {
            Ok((0..self.ndim()).into_iter().rev().collect())
        }?;

        let shape = permutation.iter().copied().map(|x| self.shape[x]).collect();
        let platform = self.platform;
        let access = platform.transpose(self.access, self.shape, permutation)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }
}

/// Unary array operations
pub trait NDArrayUnary: NDArray + Sized {
    /// The return type of a unary operation.
    type Output: Access<Self::DType>;

    /// Construct an absolute value operation.
    fn abs(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct an exponentiation operation.
    fn exp(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a natural logarithm operation.
    fn ln(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct an integer rounding operation.
    fn round(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayUnary for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseUnary<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn abs(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.abs(access))
    }

    fn exp(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.exp(access))
    }

    fn ln(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>
    where
        P: ElementwiseUnary<A, T>,
    {
        self.apply(|platform, access| platform.ln(access))
    }

    fn round(self) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.round(access))
    }
}

/// Unary boolean array operations
pub trait NDArrayUnaryBoolean: NDArray + Sized {
    /// The return type of a unary operation.
    type Output: Access<u8>;

    /// Construct a boolean not operation.
    fn not(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayUnaryBoolean for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseUnaryBoolean<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn not(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.not(access))
    }
}

/// Boolean array operations
pub trait NDArrayBoolean<O>: NDArray + Sized
where
    O: NDArray<DType = Self::DType>,
{
    type Output: Access<u8>;

    /// Construct a boolean and comparison with the `other` array.
    fn and(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean or comparison with the `other` array.
    fn or(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean xor comparison with the `other` array.
    fn xor(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, L, R, P> NDArrayBoolean<Array<T, R, P>> for Array<T, L, P>
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseBoolean<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn and(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("and", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.and(left, right))
    }

    fn or(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("or", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.or(left, right))
    }

    fn xor(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("xor", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.xor(left, right))
    }
}

/// Boolean array operations with a scalar argument
pub trait NDArrayBooleanScalar: NDArray + Sized {
    type Output: Access<u8>;

    /// Construct a boolean and operation with the `other` value.
    fn and_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean or operation with the `other` value.
    fn or_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a boolean xor operation with the `other` value.
    fn xor_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayBooleanScalar for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseBooleanScalar<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn and_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.and_scalar(access, other))
    }

    fn or_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.or_scalar(access, other))
    }

    fn xor_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.xor_scalar(access, other))
    }
}

/// Array comparison operations
pub trait NDArrayCompare<O: NDArray<DType = Self::DType>>: NDArray + Sized {
    type Output: Access<u8>;

    /// Elementwise equality comparison
    fn eq(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Elementwise greater-than-or-equal comparison
    fn ge(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Elementwise greater-than comparison
    fn gt(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Elementwise less-than-or-equal comparison
    fn le(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Elementwise less-than comparison
    fn lt(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Elementwise not-equal comparison
    fn ne(self, other: O) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, L, R, P> NDArrayCompare<Array<T, R, P>> for Array<T, L, P>
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseCompare<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn eq(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.eq(left, right))
    }

    fn ge(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.ge(left, right))
    }

    fn gt(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.gt(left, right))
    }

    fn le(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.le(left, right))
    }

    fn lt(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.lt(left, right))
    }

    fn ne(self, other: Array<T, R, P>) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        same_shape("compare", self.shape(), other.shape())?;
        self.apply_dual(other, |platform, left, right| platform.ne(left, right))
    }
}

/// Array-scalar comparison operations
pub trait NDArrayCompareScalar: NDArray + Sized {
    type Output: Access<u8>;

    /// Construct an equality comparison with the `other` value.
    fn eq_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a greater-than comparison with the `other` value.
    fn gt_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct an equal-or-greater-than comparison with the `other` value.
    fn ge_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct a less-than comparison with the `other` value.
    fn lt_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct an equal-or-less-than comparison with the `other` value.
    fn le_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Construct an not-equal comparison with the `other` value.
    fn ne_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayCompareScalar for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseScalarCompare<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn eq_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.eq_scalar(access, other))
    }

    fn gt_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.gt_scalar(access, other))
    }

    fn ge_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.ge_scalar(access, other))
    }

    fn lt_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.lt_scalar(access, other))
    }

    fn le_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.le_scalar(access, other))
    }

    fn ne_scalar(
        self,
        other: Self::DType,
    ) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.ne_scalar(access, other))
    }
}

/// Array arithmetic operations
pub trait NDArrayMath<O: NDArray<DType = Self::DType>>: NDArray + Sized {
    type Output: Access<Self::DType>;

    /// Construct an addition operation with the given `rhs`.
    fn add(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a division operation with the given `rhs`.
    fn div(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a logarithm operation with the given `base`.
    fn log(self, base: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a multiplication operation with the given `rhs`.
    fn mul(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct an operation to raise these data to the power of the given `exp`onent.
    fn pow(self, exp: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct an array subtraction operation with the given `rhs`.
    fn sub(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a modulo operation with the given `rhs`.
    fn rem(self, rhs: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, L, R, P> NDArrayMath<Array<T, R, P>> for Array<T, L, P>
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseDual<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn add(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("add", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.add(left, right))
    }

    fn div(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("div", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.div(left, right))
    }

    fn log(
        self,
        base: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("log", self.shape(), base.shape())?;
        self.apply_dual(base, |platform, left, right| platform.log(left, right))
    }

    fn mul(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("mul", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.mul(left, right))
    }

    fn pow(
        self,
        exp: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("pow", self.shape(), exp.shape())?;
        self.apply_dual(exp, |platform, left, right| platform.pow(left, right))
    }

    fn sub(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("sub", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.sub(left, right))
    }

    fn rem(
        self,
        rhs: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        same_shape("rem", self.shape(), rhs.shape())?;
        self.apply_dual(rhs, |platform, left, right| platform.rem(left, right))
    }
}

/// Array arithmetic operations with a scalar argument
pub trait NDArrayMathScalar: NDArray + Sized {
    type Output: Access<Self::DType>;

    /// Construct a scalar addition operation.
    fn add_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar division operation.
    fn div_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar logarithm operation.
    fn log_scalar(
        self,
        base: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar multiplication operation.
    fn mul_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar exponentiation operation.
    fn pow_scalar(
        self,
        exp: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar modulo operation.
    fn rem_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;

    /// Construct a scalar subtraction operation.
    fn sub_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayMathScalar for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseScalar<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn add_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.add_scalar(left, rhs))
    }

    fn div_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        if rhs != T::ZERO {
            self.apply(|platform, left| platform.div_scalar(left, rhs))
        } else {
            Err(Error::Unsupported(format!(
                "cannot divide {self:?} by {rhs}"
            )))
        }
    }

    fn log_scalar(
        self,
        base: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, arg| platform.log_scalar(arg, base))
    }

    fn mul_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.mul_scalar(left, rhs))
    }

    fn pow_scalar(
        self,
        exp: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, arg| platform.pow_scalar(arg, exp))
    }

    fn rem_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.rem_scalar(left, rhs))
    }

    fn sub_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, left| platform.sub_scalar(left, rhs))
    }
}

/// Float-specific array methods
pub trait NDArrayNumeric: NDArray + Sized
where
    Self::DType: Float,
{
    type Output: Access<u8>;

    /// Test which elements of this array are infinite.
    fn is_inf(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;

    /// Test which elements of this array are not-a-number.
    fn is_nan(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayNumeric for Array<T, A, P>
where
    T: Float,
    A: Access<T>,
    P: ElementwiseNumeric<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn is_inf(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.is_inf(access))
    }

    fn is_nan(self) -> Result<Array<u8, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.is_nan(access))
    }
}

/// Boolean array reduce operations
pub trait NDArrayReduceBoolean: NDArrayRead {
    /// Return `true` if this array contains only non-zero elements.
    fn all(self) -> Result<bool, Error>;

    /// Return `true` if this array contains any non-zero elements.
    fn any(self) -> Result<bool, Error>;
}

impl<T, A, P> NDArrayReduceBoolean for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ReduceAll<A, T>,
{
    fn all(self) -> Result<bool, Error> {
        self.platform.all(self.access)
    }

    fn any(self) -> Result<bool, Error> {
        self.platform.any(self.access)
    }
}

/// Array reduce operations
pub trait NDArrayReduceAll: NDArrayRead {
    /// Return the maximum of all elements in this array.
    fn max_all(self) -> Result<Self::DType, Error>;

    /// Return the minimum of all elements in this array.
    fn min_all(self) -> Result<Self::DType, Error>;

    /// Return the product of all elements in this array.
    fn product_all(self) -> Result<Self::DType, Error>;

    /// Return the sum of all elements in this array.
    fn sum_all(self) -> Result<Self::DType, Error>;
}

impl<'a, T, A, P> NDArrayReduceAll for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ReduceAll<A, T>,
{
    fn max_all(self) -> Result<Self::DType, Error> {
        self.platform.max(self.access)
    }

    fn min_all(self) -> Result<Self::DType, Error> {
        self.platform.min(self.access)
    }

    fn product_all(self) -> Result<Self::DType, Error> {
        self.platform.product(self.access)
    }

    fn sum_all(self) -> Result<T, Error> {
        self.platform.sum(self.access)
    }
}

impl<T, A, P> fmt::Debug for Array<T, A, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a {} array of shape {:?}",
            std::any::type_name::<T>(),
            self.shape
        )
    }
}

/// Array trigonometry methods
pub trait NDArrayTrig: NDArray + Sized {
    type Output: Access<<Self::DType as CType>::Float>;

    /// Construct a new sine operation.
    fn sin(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new arcsine operation.
    fn asin(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new hyperbolic sine operation.
    fn sinh(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new cos operation.
    fn cos(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new arccosine operation.
    fn acos(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new hyperbolic cosine operation.
    fn cosh(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new tangent operation.
    fn tan(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new arctangent operation.
    fn atan(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;

    /// Construct a new hyperbolic tangent operation.
    fn tanh(
        self,
    ) -> Result<Array<<Self::DType as CType>::Float, Self::Output, Self::Platform>, Error>;
}

impl<T, A, P> NDArrayTrig for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseTrig<A, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn sin(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.sin(access))
    }

    fn asin(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.asin(access))
    }

    fn sinh(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.sinh(access))
    }

    fn cos(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.cos(access))
    }

    fn acos(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.acos(access))
    }

    fn cosh(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.cosh(access))
    }

    fn tan(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.tan(access))
    }

    fn atan(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.atan(access))
    }

    fn tanh(self) -> Result<Array<T::Float, Self::Output, Self::Platform>, Error> {
        self.apply(|platform, access| platform.tanh(access))
    }
}

/// Conditional selection (boolean logic) methods
pub trait NDArrayWhere<T, L, R>: NDArray<DType = u8> + fmt::Debug
where
    T: CType,
{
    type Output: Access<T>;

    /// Construct a boolean selection operation.
    /// The resulting array will return values from `then` where `self` is `true`
    /// and from `or_else` where `self` is `false`.
    fn cond(self, then: L, or_else: R) -> Result<Array<T, Self::Output, Self::Platform>, Error>;
}

impl<T, A, L, R, P> NDArrayWhere<T, Array<T, L, P>, Array<T, R, P>> for Array<u8, A, P>
where
    T: CType,
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    P: GatherCond<A, L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn cond(
        self,
        then: Array<T, L, P>,
        or_else: Array<T, R, P>,
    ) -> Result<Array<T, Self::Output, Self::Platform>, Error> {
        same_shape("cond", self.shape(), then.shape())?;
        same_shape("cond", self.shape(), or_else.shape())?;

        let access = self
            .platform
            .cond(self.access, then.access, or_else.access)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

/// Matrix dual operations
pub trait MatrixDual<O>: NDArray + fmt::Debug
where
    O: NDArray<DType = Self::DType> + fmt::Debug,
{
    type Output: Access<Self::DType>;

    /// Construct an operation to multiply this matrix or batch of matrices with the `other`.
    fn matmul(self, other: O) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error>;
}

impl<T, L, R, P> MatrixDual<Array<T, R, P>> for Array<T, L, P>
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
    P: LinAlgDual<L, R, T>,
{
    type Output = AccessOp<P::Op, P>;

    fn matmul(
        self,
        other: Array<T, R, P>,
    ) -> Result<Array<Self::DType, Self::Output, Self::Platform>, Error> {
        let dims = matmul_dims(&self.shape, &other.shape).ok_or_else(|| {
            Error::Bounds(format!(
                "invalid dimensions for matrix multiply: {:?} and {:?}",
                self.shape, other.shape
            ))
        })?;

        let mut shape = Shape::with_capacity(self.ndim());
        shape.extend(self.shape.iter().rev().skip(2).rev().copied());
        shape.push(dims[1]);
        shape.push(dims[3]);

        let platform = P::select(dims.iter().product());

        let access = platform.matmul(self.access, other.access, dims)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: self.dtype,
        })
    }
}

/// Matrix unary operations
pub trait MatrixUnary: NDArray + fmt::Debug {
    type Diag: Access<Self::DType>;

    /// Construct an operation to read the diagonal(s) of this matrix or batch of matrices.
    /// This will return an error if the last two dimensions of the batch are unequal.
    fn diag(self) -> Result<Array<Self::DType, Self::Diag, Self::Platform>, Error>;
}

impl<T, A, P> MatrixUnary for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: LinAlgUnary<A, T>,
{
    type Diag = AccessOp<P::Op, P>;

    fn diag(self) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error> {
        if self.ndim() >= 2 && self.shape.last() == self.shape.iter().nth_back(1) {
            let batch_size = self.shape.iter().rev().skip(2).product();
            let dim = self.shape.last().copied().expect("dim");

            let shape = self.shape.iter().rev().skip(1).rev().copied().collect();
            let platform = P::select(batch_size * dim * dim);
            let access = platform.diag(self.access, batch_size, dim)?;

            Ok(Array {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::Bounds(format!(
                "invalid shape for diagonal: {:?}",
                self.shape
            )))
        }
    }
}

#[inline]
fn can_broadcast(left: &[usize], right: &[usize]) -> bool {
    if left.len() < right.len() {
        return can_broadcast(right, left);
    }

    for (l, r) in left.iter().copied().rev().zip(right.iter().copied().rev()) {
        if l == r || l == 1 || r == 1 {
            // pass
        } else {
            return false;
        }
    }

    true
}

#[inline]
fn matmul_dims(left: &[usize], right: &[usize]) -> Option<[usize; 4]> {
    let mut left = left.into_iter().copied().rev();
    let mut right = right.into_iter().copied().rev();

    let b = left.next()?;
    let a = left.next()?;

    let c = right.next()?;
    if right.next()? != b {
        return None;
    }

    let mut batch_size = 1;
    loop {
        match (left.next(), right.next()) {
            (Some(l), Some(r)) if l == r => {
                batch_size *= l;
            }
            (None, None) => break,
            _ => return None,
        }
    }

    Some([batch_size, a, b, c])
}

#[inline]
fn permute_for_reduce<T, A, P>(
    platform: P,
    access: A,
    shape: Shape,
    axes: Axes,
) -> Result<Accessor<T>, Error>
where
    T: CType,
    A: Access<T>,
    P: Transform<A, T>,
    Accessor<T>: From<A> + From<AccessOp<P::Transpose, P>>,
{
    let mut permutation = Axes::with_capacity(shape.len());
    permutation.extend((0..shape.len()).into_iter().filter(|x| !axes.contains(x)));
    permutation.extend(axes);

    if permutation.iter().copied().enumerate().all(|(i, x)| i == x) {
        Ok(Accessor::from(access))
    } else {
        platform
            .transpose(access, shape, permutation)
            .map(Accessor::from)
    }
}

#[inline]
fn reduce_axes(shape: &[usize], axes: &[usize], keepdims: bool) -> Result<Shape, Error> {
    let mut shape = Shape::from_slice(shape);

    for x in axes.iter().copied().rev() {
        if x >= shape.len() {
            return Err(Error::Bounds(format!(
                "axis {x} is out of bounds for {shape:?}"
            )));
        } else if keepdims {
            shape[x] = 1;
        } else {
            shape.remove(x);
        }
    }

    if shape.is_empty() {
        Ok(shape![1])
    } else {
        Ok(shape)
    }
}

#[inline]
fn same_shape(op_name: &'static str, left: &[usize], right: &[usize]) -> Result<(), Error> {
    if left == right {
        Ok(())
    } else if can_broadcast(left, right) {
        Err(Error::Bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?} (consider broadcasting)"
        )))
    } else {
        Err(Error::Bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?}"
        )))
    }
}

#[inline]
fn valid_coord(coord: &[usize], shape: &[usize]) -> Result<(), Error> {
    if coord.len() == shape.len() {
        if coord.iter().zip(shape).all(|(i, dim)| i < dim) {
            return Ok(());
        }
    }

    Err(Error::Bounds(format!(
        "invalid coordinate {coord:?} for shape {shape:?}"
    )))
}
