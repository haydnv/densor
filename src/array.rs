use std::fmt;
use std::ops::Add;

use ocl::{Buffer, OclPrm, Queue};

use super::ops::*;
use super::{
    autoqueue, broadcast_shape, kernels, AxisBound, CDatatype, Error, NDArray, NDArrayCompare,
    NDArrayCompareScalar, NDArrayRead, NDArrayReduce, Shape,
};

pub struct ArrayBase<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: OclPrm + CDatatype> ArrayBase<T> {
    pub fn concatenate(arrays: Vec<Array<T>>, axis: usize) -> Result<Self, Error> {
        todo!()
    }

    pub fn constant(shape: Shape, value: T) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![value; size],
            shape,
        }
    }

    pub fn eye(shape: Shape) -> Result<Self, Error> {
        let ndim = shape.len();
        if shape.len() < 2 || shape[ndim - 1] != shape[ndim - 2] {
            Err(Error::Bounds(format!(
                "invalid shape for identity matrix: {:?}",
                shape
            )))
        } else {
            todo!()
        }
    }

    pub fn from_vec(shape: Shape, data: Vec<T>) -> Result<Self, Error> {
        let size = shape.iter().product();
        if data.len() == size {
            Ok(Self { shape, data })
        } else {
            Err(Error::Bounds(format!(
                "{} data were provided for an array of size {}",
                data.len(),
                size
            )))
        }
    }

    pub fn random(shape: Shape) -> Result<Self, Error> {
        todo!()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: OclPrm> NDArray for ArrayBase<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'a, T: OclPrm> NDArray for &'a ArrayBase<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'a, T: OclPrm> Add for ArrayBase<T> {
    type Output = ArrayOp<ArrayAdd<Self, Self>>;

    fn add(self, rhs: Self) -> Self::Output {
        let shape = broadcast_shape(self.shape(), rhs.shape()).expect("add");
        let op = ArrayAdd::new(self, rhs);
        ArrayOp { op, shape }
    }
}

impl<'a, T: OclPrm> Add for &'a ArrayBase<T> {
    type Output = ArrayOp<ArrayAdd<Self, Self>>;

    fn add(self, rhs: Self) -> Self::Output {
        let shape = broadcast_shape(self.shape(), rhs.shape()).expect("add");
        let op = ArrayAdd::new(self, rhs);
        ArrayOp { op, shape }
    }
}

impl<T: CDatatype> NDArrayCompare<Self> for ArrayBase<T> {}

impl<'a, T: CDatatype> NDArrayCompare<&'a Self> for ArrayBase<T> {}

impl<T: CDatatype, Op: super::ops::Op<Out = T>> NDArrayCompare<ArrayOp<Op>> for ArrayBase<T> {}

impl<'a, T, Op> NDArrayCompare<&'a ArrayOp<Op>> for ArrayBase<T>
where
    T: CDatatype,
    Op: super::ops::Op<Out = T>,
{
}

impl<T: CDatatype> NDArrayCompareScalar<T> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayRead<T> for ArrayBase<T> {
    fn read(self, queue: Queue) -> Result<Buffer<T>, Error> {
        (&self).read(queue)
    }
}

impl<'a, T: CDatatype> NDArrayRead<T> for &'a ArrayBase<T> {
    fn read(self, queue: Queue) -> Result<Buffer<T>, Error> {
        let buffer = Buffer::<T>::builder()
            .queue(queue)
            .len(self.size())
            .build()
            .map_err(Error::from)?;

        buffer.write(&self.data).enq()?;

        Ok(buffer)
    }
}

impl<T: CDatatype> NDArrayReduce<T> for ArrayBase<T> {
    fn all(&self) -> Result<bool, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce_all(queue, input).map_err(Error::from)
    }

    fn all_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAll<&Self>>, Error> {
        todo!()
    }

    fn any(&self) -> Result<bool, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce_any(queue, input).map_err(Error::from)
    }

    fn any_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAny<&Self>>, Error> {
        todo!()
    }

    fn max(&self) -> Result<T, Error> {
        todo!()
    }

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMax<&Self>>, Error> {
        todo!()
    }

    fn min(&self) -> Result<T, Error> {
        todo!()
    }

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMin<&Self>>, Error> {
        todo!()
    }

    fn product(&self) -> Result<T, Error> {
        todo!()
    }

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayProduct<&Self>>, Error> {
        todo!()
    }

    fn sum(&self) -> Result<T, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce(T::zero(), "+=", queue, input, std::iter::Sum::sum).map_err(Error::from)
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArraySum<&Self>>, Error> {
        if axis >= self.ndim() {
            return Err(Error::Bounds(format!(
                "axis {} is out of bounds for {:?}",
                axis, self
            )));
        }

        let mut shape = vec![0; self.ndim() - 1];
        shape[..axis].copy_from_slice(&self.shape()[..axis]);
        shape[axis..].copy_from_slice(&self.shape()[(axis + 1)..]);

        let op = ArraySum::new(self, axis);

        Ok(ArrayOp::new(op, shape))
    }
}

impl<T: CDatatype> fmt::Debug for ArrayBase<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} array with shape {:?}", T::TYPE_STR, self.shape)
    }
}

pub struct ArrayOp<Op> {
    op: Op,
    shape: Shape,
}

impl<Op> ArrayOp<Op> {
    pub fn new(op: Op, shape: Shape) -> Self {
        Self { op, shape }
    }
}

impl<Op> NDArray for ArrayOp<Op> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'a, Op> NDArray for &'a ArrayOp<Op> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<Op: super::ops::Op> NDArrayCompareScalar<Op::Out> for ArrayOp<Op> {}

impl<Op: super::ops::Op> NDArrayRead<Op::Out> for ArrayOp<Op> {
    fn read(self, queue: Queue) -> Result<Buffer<Op::Out>, Error> {
        self.op.enqueue(queue)
    }
}

impl<'a, Op: super::ops::Op> NDArrayRead<Op::Out> for &'a ArrayOp<Op> {
    fn read(self, queue: Queue) -> Result<Buffer<Op::Out>, Error> {
        self.op.enqueue(queue)
    }
}

impl<Op: super::ops::Op> NDArrayReduce<Op::Out> for ArrayOp<Op> {
    fn all(&self) -> Result<bool, Error> {
        let queue = autoqueue(None)?;
        let input = self.op.enqueue(queue.clone())?;
        kernels::reduce_all(queue, input).map_err(Error::from)
    }

    fn all_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAll<&Self>>, Error> {
        todo!()
    }

    fn any(&self) -> Result<bool, Error> {
        let queue = autoqueue(None)?;
        let input = self.op.enqueue(queue.clone())?;
        kernels::reduce_any(queue, input).map_err(Error::from)
    }

    fn any_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAny<&Self>>, Error> {
        todo!()
    }

    fn max(&self) -> Result<Op::Out, Error> {
        todo!()
    }

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMax<&Self>>, Error> {
        todo!()
    }

    fn min(&self) -> Result<Op::Out, Error> {
        todo!()
    }

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMin<&Self>>, Error> {
        todo!()
    }

    fn product(&self) -> Result<Op::Out, Error> {
        todo!()
    }

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayProduct<&Self>>, Error> {
        todo!()
    }

    fn sum(&self) -> Result<Op::Out, Error> {
        todo!()
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArraySum<&Self>>, Error> {
        todo!()
    }
}

impl<Op: super::ops::Op> fmt::Debug for ArrayOp<Op> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} array result with shape {:?}",
            Op::Out::TYPE_STR,
            self.shape
        )
    }
}

pub struct ArraySlice<A> {
    source: A,
    bounds: Vec<AxisBound>,
    shape: Shape,
}

impl<A> NDArray for ArraySlice<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub struct ArrayView<A> {
    source: A,
    strides: Vec<u64>,
    shape: Shape,
}

impl<A> NDArray for ArrayView<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub enum Array<T: OclPrm> {
    Base(ArrayBase<T>),
    Slice(ArraySlice<Box<Self>>),
    View(ArrayView<Box<Self>>),
    Op(ArrayOp<Box<dyn super::ops::Op<Out = T>>>),
}

impl<T: OclPrm> NDArray for Array<T> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Base(base) => base.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::View(view) => view.shape(),
            Self::Op(op) => op.shape(),
        }
    }
}
