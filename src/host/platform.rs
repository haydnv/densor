use rayon::prelude::*;

use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
use crate::host::StackVec;
use crate::ops::{
    Construct, ElementwiseBoolean, ElementwiseCompare, ElementwiseDual, ElementwiseScalarCompare,
    ElementwiseUnary, LinAlgDual, Random, ReduceAll, ReduceAxis, Transform,
};
use crate::platform::{Convert, PlatformInstance};
use crate::{stackvec, Axes, CType, Constant, Error, Float, Range, Shape};

use super::buffer::{Buffer, SliceConverter};
use super::ops::*;

pub const VEC_MIN_SIZE: usize = 64;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<T: CType> Constant<T> for Stack {
    type Buffer = StackVec<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        Ok(stackvec![value; size])
    }
}

impl<A, T> ReduceAll<A, T> for Stack
where
    A: Access<T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().all(|n| n != T::ZERO))
    }

    fn any(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().any(|n| n != T::ZERO))
    }

    fn max(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().reduce(T::max).expect("max"))
    }

    fn min(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().reduce(T::min).expect("min"))
    }

    fn product(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().product())
    }

    fn sum(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().sum())
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Heap;

impl PlatformInstance for Heap {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<T: CType> Constant<T> for Heap {
    type Buffer = Vec<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        Ok(vec![value; size])
    }
}

impl<A, T> ReduceAll<A, T> for Heap
where
    A: Access<T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().all(|n| n != T::ZERO))
    }

    fn any(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().any(|n| n != T::ZERO))
    }

    fn max(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().reduce(|| T::MIN, T::max))
    }

    fn min(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().reduce(|| T::MAX, T::min))
    }

    fn product(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().product())
    }

    fn sum(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().sum())
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Host {
    Stack(Stack),
    Heap(Heap),
}

impl PlatformInstance for Host {
    fn select(size_hint: usize) -> Self {
        if size_hint < VEC_MIN_SIZE {
            Self::Stack(Stack)
        } else {
            Self::Heap(Heap)
        }
    }
}

impl<T: CType> Constant<T> for Host {
    type Buffer = Buffer<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        match self {
            Self::Heap(heap) => heap.constant(value, size).map(Buffer::Heap),
            Self::Stack(stack) => stack.constant(value, size).map(Buffer::Stack),
        }
    }
}

impl<'a, T: CType> Convert<'a, T> for Host {
    type Buffer = SliceConverter<'a, T>;

    fn convert(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error> {
        buffer.to_slice()
    }
}

impl From<Heap> for Host {
    fn from(heap: Heap) -> Self {
        Self::Heap(heap)
    }
}

impl From<Stack> for Host {
    fn from(stack: Stack) -> Self {
        Self::Stack(stack)
    }
}

impl<T: CType> Construct<T> for Host {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        if start <= stop {
            let step = (stop - start).to_float().to_f64() / size as f64;
            Ok(Linear::new(start, step, size).into())
        } else {
            Err(Error::Bounds(format!("invalid range: [{start}, {stop})")))
        }
    }
}

impl<L, R, T> ElementwiseBoolean<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Dual<L, R, T, u8>;

    fn and(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::and(left, right).into())
    }

    fn or(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::or(left, right).into())
    }

    fn xor(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::xor(left, right).into())
    }
}

impl<L, R, T> ElementwiseCompare<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Dual<L, R, T, u8>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::eq(left, right).into())
    }

    fn ge(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::ge(left, right).into())
    }

    fn gt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::gt(left, right).into())
    }

    fn le(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::le(left, right).into())
    }

    fn lt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::lt(left, right).into())
    }

    fn ne(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::ne(left, right).into())
    }
}

impl<A: Access<T>, T: CType> ElementwiseScalarCompare<A, T> for Host {
    type Op = Scalar<A, T, u8>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::eq(left, right).into())
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::ge(left, right).into())
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::gt(left, right).into())
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::le(left, right).into())
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::lt(left, right).into())
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::ne(left, right).into())
    }
}

impl<L, R, T> ElementwiseDual<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Dual<L, R, T, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::add(left, right).into())
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::sub(left, right).into())
    }
}

impl<A: Access<T>, T: CType> ElementwiseUnary<A, T> for Host {
    type Op = Unary<A, T, T>;

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::ln(access).into())
    }
}

impl<L, R, T> LinAlgDual<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = MatMul<L, R, T>;

    fn matmul(
        self,
        left: L,
        right: R,
        dims: [usize; 4],
    ) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(MatMul::new(left, right, dims).into())
    }
}

impl Random for Host {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        Ok(RandomNormal::new(size).into())
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        Ok(RandomUniform::new(size).into())
    }
}

impl<A: Access<T>, T: CType> ReduceAll<A, T> for Host {
    fn all(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Heap(heap) => heap.all(access),
            Self::Stack(stack) => stack.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Heap(heap) => heap.any(access),
            Self::Stack(stack) => stack.any(access),
        }
    }

    fn max(self, access: A) -> Result<T, Error> {
        match self {
            Self::Heap(heap) => heap.max(access),
            Self::Stack(stack) => stack.max(access),
        }
    }

    fn min(self, access: A) -> Result<T, Error> {
        match self {
            Self::Heap(heap) => heap.min(access),
            Self::Stack(stack) => stack.min(access),
        }
    }

    fn product(self, access: A) -> Result<T, Error> {
        match self {
            Self::Heap(heap) => heap.product(access),
            Self::Stack(stack) => stack.product(access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match self {
            Self::Heap(heap) => heap.sum(access),
            Self::Stack(stack) => stack.sum(access),
        }
    }
}

impl<A: Access<T>, T: CType> ReduceAxis<A, T> for Host {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Reduce::max(access, stride).into())
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Reduce::min(access, stride).into())
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Reduce::product(access, stride).into())
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Reduce::sum(access, stride).into())
    }
}

impl<'a, A, T> Transform<A, T> for Host
where
    A: Access<T>,
    T: CType,
{
    type Broadcast = View<A, T>;
    type Slice = Slice<A, T>;
    type Transpose = View<A, T>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        Ok(View::broadcast(access, shape, broadcast).into())
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        Ok(Slice::new(access, shape, range).into())
    }

    fn transpose(
        self,
        access: A,
        shape: Shape,
        permutation: Axes,
    ) -> Result<AccessOp<Self::Transpose, Self>, Error> {
        Ok(View::transpose(access, shape, permutation).into())
    }
}
