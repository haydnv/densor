use std::ops::{Add, Mul};

use ha_ndarray::{
    ArrayBase, Error, MatrixMath, NDArray, NDArrayCompareScalar, NDArrayRead, NDArrayReduce,
};

#[test]
fn test_matmul() -> Result<(), Error> {
    let left = ArrayBase::constant(vec![2, 3], 1.);
    let right = ArrayBase::constant(vec![3, 4], 1.);

    let expected = ArrayBase::from_vec(
        vec![2, 4],
        [[3., 3., 3., 3.], [3., 3., 3., 3.]]
            .into_iter()
            .flatten()
            .collect(),
    );

    let actual = left.matmul(&right)?;
    assert_eq!(actual.shape(), [2, 4]);
    assert!(actual.eq(3.)?.all()?);
    Ok(())
}
