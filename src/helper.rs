//! Various helper functions for managing collections.

/// Deletes from `vec` only the element `i` such that the corresponding `mask[i]` is `true`.
pub fn filter_from_mask<T: Copy>(mask: &[bool], vec: &mut Vec<T>) {
    let mut i = 0;

    vec.retain(|_| {
        let delete = mask[i];
        i += 1;
        !delete
    })
}
