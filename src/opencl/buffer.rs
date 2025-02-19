use std::ops::Deref;

use ocl::Buffer;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::opencl::OpenCL;
use crate::{CType, Error};

impl<T: CType> BufferInstance<T> for Buffer<T> {
    fn read(&self) -> BufferConverter<T> {
        BufferConverter::CL(self.into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        if offset < self.len() {
            let slice = self.map().offset(offset).len(1).read();
            let value = unsafe { slice.enq()? };
            Ok(value.get(0).copied().expect("value"))
        } else {
            Err(Error::Bounds(format!(
                "invalid offset {offset} for a buffer of length {}",
                self.len()
            )))
        }
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: CType> BufferMut<T> for Buffer<T> {
    fn cl(&mut self) -> Result<&mut Buffer<T>, Error> {
        Ok(self)
    }

    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        if data.len() == self.len() {
            let data = data.to_cl()?;
            data.copy(self, None, None).enq().map_err(Error::from)
        } else {
            Err(Error::Bounds(format!(
                "cannot overwrite a buffer of size {} with one of size {}",
                self.len(),
                data.len()
            )))
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        let buf = Buffer::builder()
            .context(OpenCL::context())
            .len(self.len())
            .fill_val(value)
            .build()?;

        *self = buf;
        Ok(())
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        if offset < self.len() {
            let slice = self.map().offset(offset).len(1).read();
            let mut slice = unsafe { slice.enq()? };
            slice.as_mut()[0] = value;
            Ok(())
        } else {
            Err(Error::Bounds(format!(
                "invalid offset {offset} for a buffer of length {}",
                self.len()
            )))
        }
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a Buffer<T> {
    fn read(&self) -> BufferConverter<T> {
        BufferConverter::CL((*self).into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(*self, offset)
    }

    fn len(&self) -> usize {
        Buffer::len(self)
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a mut Buffer<T> {
    fn read(&self) -> BufferConverter<T> {
        BufferConverter::CL((&**self).into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(*self, offset)
    }

    fn len(&self) -> usize {
        Buffer::len(self)
    }
}

impl<'a, T: CType> BufferMut<T> for &'a mut Buffer<T> {
    fn cl(&mut self) -> Result<&mut Buffer<T>, Error> {
        Ok(*self)
    }

    fn write<'b>(&mut self, data: BufferConverter<'b, T>) -> Result<(), Error> {
        BufferMut::write(&mut **self, data.into())
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        BufferMut::write_value(&mut **self, value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        BufferMut::write_value_at(&mut **self, offset, value)
    }
}

/// A buffer in OpenCL memory
#[derive(Clone)]
pub enum CLConverter<'a, T: CType> {
    Owned(Buffer<T>),
    Borrowed(&'a Buffer<T>),
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> CLConverter<'a, T> {
    /// Return this buffer as an owned [`Buffer`].
    /// This will allocate a new [`Buffer`] only if this buffer is borrowed.
    pub fn into_buffer(self) -> Result<Buffer<T>, Error> {
        match self {
            Self::Owned(buffer) => Ok(buffer),
            Self::Borrowed(buffer) => {
                let cl_queue = buffer.default_queue().expect("OpenCL queue");
                let mut copy = Buffer::builder()
                    .queue(cl_queue.clone())
                    .len(buffer.len())
                    .build()?;

                buffer.copy(&mut copy, None, None).enq()?;

                Ok(copy)
            }
        }
    }

    /// Return the number of elements in this buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(buffer) => buffer.len(),
            Self::Borrowed(buffer) => buffer.len(),
        }
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> Deref for CLConverter<'a, T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Buffer<T> {
        match self {
            Self::Owned(buffer) => &buffer,
            Self::Borrowed(buffer) => buffer,
        }
    }
}

impl<T: CType> From<Buffer<T>> for CLConverter<'static, T> {
    fn from(buf: Buffer<T>) -> Self {
        Self::Owned(buf)
    }
}

impl<'a, T: CType> From<&'a Buffer<T>> for CLConverter<'a, T> {
    fn from(buf: &'a Buffer<T>) -> Self {
        Self::Borrowed(buf)
    }
}
