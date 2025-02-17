use std::fmt;

#[cfg(feature = "stream")]
use destream::{de, en};
use get_size::GetSize;

#[cfg(feature = "opencl")]
use crate::opencl;
use crate::{host, CType, Error};

/// A data buffer
pub trait BufferInstance<T: CType>: Send + Sync {
    /// Borrow this buffer as a [`BufferConverter`].
    fn read(&self) -> BufferConverter<T>;

    /// Read a single value in this buffer.
    fn read_value(&self, offset: usize) -> Result<T, Error>;

    /// Return the length of this buffer.
    fn len(&self) -> usize;
}

/// A mutable data buffer
pub trait BufferMut<T: CType>: BufferInstance<T> + fmt::Debug {
    #[cfg(feature = "opencl")]
    /// Borrow this buffer as an [`ocl::Buffer`], or return an error if this not an OpenCL buffer.
    fn cl(&mut self) -> Result<&mut ocl::Buffer<T>, Error> {
        Err(Error::Unsupported(format!(
            "not an OpenCL buffer: {self:?}"
        )))
    }

    /// Overwrite this buffer.
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error>;

    /// Overwrite this buffer with a single value.
    fn write_value(&mut self, value: T) -> Result<(), Error>;

    /// Overwrite a single value in this buffer.
    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error>;
}

/// A general-purpose buffer which can represent a buffer on any supported platform.
#[derive(Clone)]
pub enum Buffer<T: CType> {
    #[cfg(feature = "opencl")]
    CL(ocl::Buffer<T>),
    Host(host::Buffer<T>),
}

impl<T: CType> Buffer<T> {
    /// Construct a new [`Buffer`] from a slice of data.
    pub fn from_slice(slice: &[T]) -> Result<Self, Error> {
        BufferConverter::from(slice).into_buffer()
    }
}

impl<T: CType> GetSize for Buffer<T> {
    fn get_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<T: CType> BufferInstance<T> for Buffer<T> {
    fn read(&self) -> BufferConverter<T> {
        BufferConverter::from(self)
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.read_value(offset),
            Self::Host(buf) => buf.read_value(offset),
        }
    }

    fn len(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.len(),
            Self::Host(buf) => buf.len(),
        }
    }
}

impl<T: CType> BufferMut<T> for Buffer<T> {
    #[cfg(feature = "opencl")]
    fn cl(&mut self) -> Result<&mut ocl::Buffer<T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.cl(),
            Self::Host(buf) => buf.cl(),
        }
    }

    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.write(data),
            Self::Host(buf) => buf.write(data),
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.write_value(value),
            Self::Host(buf) => buf.write_value(value),
        }
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.write_value_at(offset, value),
            Self::Host(buf) => buf.write_value_at(offset, value),
        }
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a Buffer<T> {
    fn read(&self) -> BufferConverter<T> {
        BufferConverter::from(*self)
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(*self, offset)
    }

    fn len(&self) -> usize {
        BufferInstance::len(*self)
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a mut Buffer<T> {
    fn read(&self) -> BufferConverter<T> {
        BufferConverter::from(&**self)
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(&**self, offset)
    }

    fn len(&self) -> usize {
        BufferInstance::len(*self)
    }
}

impl<'a, T: CType> BufferMut<T> for &'a mut Buffer<T> {
    #[cfg(feature = "opencl")]
    fn cl(&mut self) -> Result<&mut ocl::Buffer<T>, Error> {
        Buffer::<T>::cl(&mut **self)
    }

    fn write<'b>(&mut self, data: BufferConverter<'b, T>) -> Result<(), Error> {
        Buffer::<T>::write(*self, data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        Buffer::<T>::write_value(*self, value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        Buffer::<T>::write_value_at(*self, offset, value)
    }
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CType> BufferInstance<T> for freqfs::FileReadGuardOwned<FE, Buffer<T>> {
    fn read(&self) -> BufferConverter<T> {
        BufferInstance::read(&**self)
    }

    fn len(&self) -> usize {
        BufferInstance::len(&**self)
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(&**self, offset)
    }
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CType> BufferInstance<T> for freqfs::FileWriteGuardOwned<FE, Buffer<T>> {
    fn read(&self) -> BufferConverter<T> {
        BufferInstance::read(&**self)
    }

    fn len(&self) -> usize {
        BufferInstance::len(&**self)
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        BufferInstance::read_value(&**self, offset)
    }
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CType> BufferMut<T> for freqfs::FileWriteGuardOwned<FE, Buffer<T>> {
    #[cfg(feature = "opencl")]
    fn cl(&mut self) -> Result<&mut ocl::Buffer<T>, Error> {
        BufferMut::cl(&mut **self)
    }

    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        BufferMut::write(&mut **self, data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        BufferMut::write_value(&mut **self, value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        BufferMut::write_value_at(&mut **self, offset, value)
    }
}

#[cfg(feature = "opencl")]
impl<T: CType> From<ocl::Buffer<T>> for Buffer<T> {
    fn from(buf: ocl::Buffer<T>) -> Self {
        Self::CL(buf)
    }
}

impl<T: CType> From<host::StackVec<T>> for Buffer<T> {
    fn from(buf: host::StackVec<T>) -> Self {
        Self::Host(buf.into())
    }
}

impl<T: CType> From<Vec<T>> for Buffer<T> {
    fn from(buf: Vec<T>) -> Self {
        Self::Host(buf.into())
    }
}

impl<T: CType> From<host::Buffer<T>> for Buffer<T> {
    fn from(buf: host::Buffer<T>) -> Self {
        Self::Host(buf)
    }
}

#[derive(Clone)]
/// A sequence of elements in a single contiguous block of memory
pub enum BufferConverter<'a, T: CType> {
    #[cfg(feature = "opencl")]
    CL(opencl::CLConverter<'a, T>),
    Host(host::SliceConverter<'a, T>),
}

impl<'a, T: CType> BufferConverter<'a, T> {
    /// Return an owned [`Buffer`], allocating memory only if this [`BufferConverter`]'s data is borrowed.
    pub fn into_buffer(self) -> Result<Buffer<T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.into_buffer().map(Buffer::CL),
            Self::Host(buffer) => Ok(Buffer::Host(buffer.into_buffer())),
        }
    }

    /// Return the number of elements in this [`Buffer`].
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.len(),
            Self::Host(buffer) => buffer.len(),
        }
    }

    #[cfg(feature = "opencl")]
    /// Ensure that this [`Buffer`] is in OpenCL memory by making a copy if necessary.
    pub fn to_cl(self) -> Result<opencl::CLConverter<'a, T>, ocl::Error> {
        match self {
            Self::CL(buffer) => Ok(buffer),
            Self::Host(buffer) => {
                opencl::OpenCL::copy_into_buffer(buffer.as_ref()).map(opencl::CLConverter::Owned)
            }
        }
    }

    /// Ensure that this buffer is in host memory by making a copy if necessary.
    #[inline]
    pub fn to_slice(self) -> Result<host::SliceConverter<'a, T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => {
                let mut copy = vec![T::default(); buffer.len()];
                buffer.read(&mut copy[..]).enq()?;
                Ok(host::SliceConverter::from(copy))
            }
            Self::Host(buffer) => Ok(buffer),
        }
    }
}

impl<T: CType> From<Buffer<T>> for BufferConverter<'static, T> {
    fn from(buf: Buffer<T>) -> Self {
        match buf {
            #[cfg(feature = "opencl")]
            Buffer::CL(buf) => Self::CL(buf.into()),
            Buffer::Host(buf) => Self::Host(buf.into()),
        }
    }
}

impl<'a, T: CType> From<&'a Buffer<T>> for BufferConverter<'a, T> {
    fn from(buf: &'a Buffer<T>) -> Self {
        match buf {
            #[cfg(feature = "opencl")]
            Buffer::CL(buf) => Self::CL(buf.into()),
            Buffer::Host(buf) => Self::Host(buf.into()),
        }
    }
}

impl<T: CType> From<Vec<T>> for BufferConverter<'static, T> {
    fn from(buf: Vec<T>) -> Self {
        Self::Host(buf.into())
    }
}

impl<T: CType> From<host::StackVec<T>> for BufferConverter<'static, T> {
    fn from(buf: host::StackVec<T>) -> Self {
        Self::Host(buf.into())
    }
}

impl<T: CType> From<host::Buffer<T>> for BufferConverter<'static, T> {
    fn from(buf: host::Buffer<T>) -> Self {
        Self::Host(buf.into())
    }
}

impl<'a, T: CType> From<&'a [T]> for BufferConverter<'a, T> {
    fn from(buf: &'a [T]) -> Self {
        Self::Host(buf.into())
    }
}

#[cfg(feature = "opencl")]
impl<T: CType> From<ocl::Buffer<T>> for BufferConverter<'static, T> {
    fn from(buf: ocl::Buffer<T>) -> Self {
        Self::CL(buf.into())
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> From<&'a ocl::Buffer<T>> for BufferConverter<'a, T> {
    fn from(buf: &'a ocl::Buffer<T>) -> Self {
        Self::CL(buf.into())
    }
}

#[cfg(feature = "stream")]
struct BufferVisitor<T> {
    data: Vec<T>,
}

#[cfg(feature = "stream")]
impl<T> BufferVisitor<T> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }
}

#[cfg(feature = "stream")]
macro_rules! decode_buffer {
    ($t:ty, $name:expr, $decode:ident, $visit:ident, $encode:ident) => {
        impl de::Visitor for BufferVisitor<$t> {
            type Value = Buffer<$t>;

            fn expecting() -> &'static str {
                $name
            }

            async fn $visit<A: de::ArrayAccess<$t>>(
                self,
                mut array: A,
            ) -> Result<Self::Value, A::Error> {
                use crate::platform::{Convert, PlatformInstance};

                const BUF_SIZE: usize = 4_096;
                let mut data = self.data;

                let mut buf = [<$t>::ZERO; BUF_SIZE];
                loop {
                    let len = array.buffer(&mut buf).await?;
                    if len == 0 {
                        break;
                    } else {
                        data.extend_from_slice(&buf[..len]);
                    }
                }

                crate::Platform::select(data.len())
                    .convert(data.into())
                    .map_err(de::Error::custom)
            }
        }

        impl de::FromStream for Buffer<$t> {
            type Context = ();

            async fn from_stream<D: de::Decoder>(
                _cxt: (),
                decoder: &mut D,
            ) -> Result<Self, D::Error> {
                decoder.$decode(BufferVisitor::<$t>::new()).await
            }
        }

        impl<'en> en::ToStream<'en> for Buffer<$t> {
            fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
                match self {
                    Self::Host(buffer) => {
                        let fut = futures::future::ready(buffer.to_vec());
                        let stream = futures::stream::once(fut);
                        encoder.$encode(stream)
                    }
                    #[cfg(feature = "opencl")]
                    Self::CL(buffer) => {
                        let mut data = Vec::with_capacity(buffer.len());
                        buffer.read(&mut data).enq().map_err(en::Error::custom)?;
                        encoder.$encode(futures::stream::once(futures::future::ready(data)))
                    }
                }
            }
        }

        impl<'en> en::IntoStream<'en> for Buffer<$t> {
            fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
                match self {
                    Self::Host(buffer) => {
                        let buffer = buffer.to_vec();
                        encoder.$encode(futures::stream::once(futures::future::ready(buffer)))
                    }
                    #[cfg(feature = "opencl")]
                    Self::CL(buffer) => {
                        let mut data = Vec::with_capacity(buffer.len());
                        buffer.read(&mut data).enq().map_err(en::Error::custom)?;
                        encoder.$encode(futures::stream::once(futures::future::ready(data)))
                    }
                }
            }
        }
    };
}

#[cfg(feature = "stream")]
decode_buffer!(
    u8,
    "byte array",
    decode_array_u8,
    visit_array_u8,
    encode_array_u8
);

#[cfg(feature = "stream")]
decode_buffer!(
    u16,
    "16-bit unsigned int array",
    decode_array_u16,
    visit_array_u16,
    encode_array_u16
);

#[cfg(feature = "stream")]
decode_buffer!(
    u32,
    "32-bit unsigned int array",
    decode_array_u32,
    visit_array_u32,
    encode_array_u32
);

#[cfg(feature = "stream")]
decode_buffer!(
    u64,
    "64-bit unsigned int array",
    decode_array_u64,
    visit_array_u64,
    encode_array_u64
);

#[cfg(feature = "stream")]
decode_buffer!(
    i16,
    "16-bit int array",
    decode_array_i16,
    visit_array_i16,
    encode_array_i16
);

#[cfg(feature = "stream")]
decode_buffer!(
    i32,
    "32-bit int array",
    decode_array_i32,
    visit_array_i32,
    encode_array_i32
);

#[cfg(feature = "stream")]
decode_buffer!(
    i64,
    "64-bit int array",
    decode_array_i64,
    visit_array_i64,
    encode_array_i64
);

#[cfg(feature = "stream")]
decode_buffer!(
    f32,
    "32-bit int array",
    decode_array_f32,
    visit_array_f32,
    encode_array_f32
);

#[cfg(feature = "stream")]
decode_buffer!(
    f64,
    "64-bit int array",
    decode_array_f64,
    visit_array_f64,
    encode_array_f64
);

impl<T: CType + fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Host(buffer) => fmt::Debug::fmt(buffer, f),
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => fmt::Debug::fmt(buffer, f),
        }
    }
}
