use std::marker::PhantomData;

use ocl::{Buffer, Error, OclPrm, Queue};

use super::CDatatype;

pub trait Op<Out: OclPrm> {
    fn enqueue(&self, queue: Queue, output: Option<Buffer<Out>>) -> Result<Buffer<Out>, Error>;
}

// constructors

pub struct ArrayConstant<T> {
    value: T,
    size: u64,
}

pub struct ArrayRandom {
    size: u64,
}

pub struct MatEye {
    count: u64,
    size: u64,
}

// arithmetic

pub struct ArrayAdd<L, R> {
    left: L,
    right: R,
}

pub struct ArrayDiv<L, R> {
    left: L,
    right: R,
}

pub struct ArrayMul<L, R> {
    left: L,
    right: R,
}

pub struct ArrayMod<L, R> {
    left: L,
    right: R,
}

pub struct ArraySub<L, R> {
    left: L,
    right: R,
}

// linear algebra

pub struct MatDiag<A> {
    source: A,
}

pub struct MatMul<L, R> {
    left: L,
    right: R,
}

// comparison

pub struct ArrayEq<L, R> {
    left: L,
    right: R,
}

pub struct ArrayGT<L, R> {
    left: L,
    right: R,
}

pub struct ArrayGTE<L, R> {
    left: L,
    right: R,
}

pub struct ArrayLT<L, R> {
    left: L,
    right: R,
}

pub struct ArrayLTE<L, R> {
    left: L,
    right: R,
}

pub struct ArrayNE<L, R> {
    left: L,
    right: R,
}

// reduction

pub struct ArrayMax<A> {
    source: A,
}

pub struct ArrayMin<A> {
    source: A,
}

pub struct ArrayProduct<A> {
    source: A,
}

pub struct ArraySum<A> {
    source: A,
}

// other unary ops

pub struct ArrayCast<A, O> {
    source: A,
    dtype: PhantomData<O>,
}
