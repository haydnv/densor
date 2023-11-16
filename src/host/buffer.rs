use std::borrow::Borrow;
use std::ops::Deref;

use smallvec::SmallVec;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::{CType, Error};

use super::VEC_MIN_SIZE;

pub type StackVec<T> = SmallVec<[T; VEC_MIN_SIZE]>;

impl<T: CType> BufferInstance<T> for StackVec<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(self.as_slice().into())
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferMut<'a, T> for StackVec<T> {
    type Data = &'a [T];

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.as_mut_slice().write(data)
    }
}

impl<T: CType> BufferInstance<T> for Vec<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(self.as_slice().into())
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferMut<'a, T> for Vec<T> {
    type Data = &'a [T];

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.as_mut_slice().write(data)
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a [T] {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok((*self).into())
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a mut [T] {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok((&**self).into())
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferMut<'a, T> for &'a mut [T] {
    type Data = &'a [T];

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        if data.len() == self.len() {
            self.copy_from_slice(data);
            Ok(())
        } else {
            Err(Error::Bounds(format!(
                "cannot overwrite a buffer of size {} with one of size {}",
                self.len(),
                data.len()
            )))
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Buffer<T> {
    Heap(Vec<T>),
    Stack(StackVec<T>),
}

impl<T: Clone> Buffer<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Heap(data) => data,
            Self::Stack(data) => data.into_vec(),
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        match self {
            Self::Heap(data) => data.to_vec(),
            Self::Stack(data) => data.to_vec(),
        }
    }
}

impl<T> Borrow<[T]> for Buffer<T> {
    fn borrow(&self) -> &[T] {
        match self {
            Self::Heap(buf) => buf.borrow(),
            Self::Stack(buf) => buf.borrow(),
        }
    }
}

impl<T> AsMut<[T]> for Buffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        match self {
            Self::Heap(buf) => buf.as_mut_slice(),
            Self::Stack(buf) => buf.as_mut_slice(),
        }
    }
}

impl<T: CType> BufferInstance<T> for Buffer<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::Host(self.into()))
    }

    fn size(&self) -> usize {
        match self {
            Self::Heap(buf) => buf.size(),
            Self::Stack(buf) => buf.size(),
        }
    }
}

impl<'a, T: CType> BufferMut<'a, T> for Buffer<T> {
    type Data = SliceConverter<'a, T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        match self {
            Self::Heap(buf) => buf.write(data.as_ref()),
            Self::Stack(buf) => buf.write(data.as_ref()),
        }
    }
}

impl<T> From<StackVec<T>> for Buffer<T> {
    fn from(buf: StackVec<T>) -> Self {
        Self::Stack(buf)
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    fn from(buf: Vec<T>) -> Self {
        Self::Heap(buf)
    }
}

#[derive(Clone)]
/// A buffer in host memory, either borrowed or owned
pub enum SliceConverter<'a, T> {
    Heap(Vec<T>),
    Stack(StackVec<T>),
    Slice(&'a [T]),
}

impl<'a, T> SliceConverter<'a, T> {
    /// Return the number of elements in this buffer.
    pub fn size(&self) -> usize {
        match self {
            Self::Heap(vec) => vec.len(),
            Self::Stack(vec) => vec.len(),
            Self::Slice(slice) => slice.len(),
        }
    }
}

impl<'a, T: Copy> SliceConverter<'a, T> {
    /// Return this buffer as an owned [`Vec`].
    /// This will allocate a new [`Vec`] if this buffer is a [`StackVec`] or borrowed slice.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Heap(vec) => vec,
            Self::Stack(vec) => vec.into_vec(),
            Self::Slice(slice) => slice.to_vec(),
        }
    }

    /// Return this buffer as an owned [`StackVec`].
    pub fn into_stackvec(self) -> StackVec<T> {
        match self {
            Self::Heap(vec) => vec.into(),
            Self::Stack(vec) => vec,
            Self::Slice(slice) => StackVec::from_slice(slice),
        }
    }

    /// Return this buffer as an owned host [`Buffer`].
    pub fn into_buffer(self) -> Buffer<T> {
        match self {
            Self::Heap(vec) => Buffer::Heap(vec),
            Self::Stack(vec) => Buffer::Stack(vec),
            Self::Slice(slice) => {
                if slice.len() < VEC_MIN_SIZE {
                    Buffer::Stack(StackVec::from_slice(slice))
                } else {
                    Buffer::Heap(slice.to_vec())
                }
            }
        }
    }
}

impl<T> From<Buffer<T>> for SliceConverter<'static, T> {
    fn from(buf: Buffer<T>) -> Self {
        match buf {
            Buffer::Heap(buf) => SliceConverter::Heap(buf),
            Buffer::Stack(buf) => SliceConverter::Stack(buf),
        }
    }
}

impl<'a, T> From<&'a Buffer<T>> for SliceConverter<'a, T> {
    fn from(buf: &'a Buffer<T>) -> Self {
        match buf {
            Buffer::Heap(slice) => slice.as_slice().into(),
            Buffer::Stack(slice) => slice.as_slice().into(),
        }
    }
}

impl<T> From<StackVec<T>> for SliceConverter<'static, T> {
    fn from(vec: StackVec<T>) -> Self {
        Self::Stack(vec)
    }
}

impl<T> From<Vec<T>> for SliceConverter<'static, T> {
    fn from(vec: Vec<T>) -> Self {
        Self::Heap(vec)
    }
}

impl<'a, T> From<&'a [T]> for SliceConverter<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self::Slice(slice)
    }
}

impl<'a, T> Deref for SliceConverter<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Heap(data) => data.as_slice(),
            Self::Stack(data) => data.as_slice(),
            Self::Slice(slice) => slice,
        }
    }
}
