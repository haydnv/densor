use ha_ndarray::{ArrayBase, Error, NDArrayCompare, NDArrayReduce, NDArrayTransform};

#[test]
fn test_add() -> Result<(), Error> {
    let shape = vec![5, 2];
    let left = ArrayBase::from_vec(shape.to_vec(), (0..10).into_iter().collect())?;
    let right = ArrayBase::from_vec(shape.to_vec(), (0..10).into_iter().rev().collect())?;
    let actual = left + right;
    let expected = ArrayBase::constant(shape, 9);
    assert!(expected.eq(&actual)?.all()?);
    Ok(())
}

#[test]
fn test_expand_and_broadcast_and_sub() -> Result<(), Error> {
    let left = ArrayBase::from_vec(vec![2, 3], (0i32..6).into_iter().collect())?;
    let right = ArrayBase::from_vec(vec![2], vec![0, 1])?;
    let expected = ArrayBase::from_vec(vec![2, 3], vec![0, 1, 2, 2, 3, 4])?;
    let actual = left - right.expand_dims(vec![1])?.broadcast(vec![2, 3])?;
    assert!(expected.eq(&actual)?.all()?);
    Ok(())
}
