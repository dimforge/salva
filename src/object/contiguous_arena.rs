use generational_arena::{Arena, Index};
use na::RealField;
use std::marker::PhantomData;

pub type ContiguousArenaIndex = Index;

#[derive(Clone)]
pub struct ContiguousArena<Idx, T> {
    objects: Vec<T>,
    rev_indices: Vec<Index>,
    indices: Arena<usize>,
    phantoms: PhantomData<Idx>,
}

impl<Idx, T> ContiguousArena<Idx, T> {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            indices: Arena::new(),
            rev_indices: Vec::new(),
            phantoms: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    #[inline]
    pub fn get(&self, handle: Idx) -> Option<&T>
    where
        Idx: Into<ContiguousArenaIndex>,
    {
        self.objects.get(*self.indices.get(handle.into())?)
    }

    #[inline]
    pub fn get_mut(&mut self, handle: Idx) -> Option<&mut T>
    where
        Idx: Into<ContiguousArenaIndex>,
    {
        self.objects.get_mut(*self.indices.get(handle.into())?)
    }

    #[inline]
    pub fn values(&self) -> std::slice::Iter<T> {
        self.objects.iter()
    }

    #[inline]
    pub fn values_mut(&mut self) -> std::slice::IterMut<T> {
        self.objects.iter_mut()
    }

    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Idx, &'a T)> + 'a
    where
        Idx: From<ContiguousArenaIndex>,
    {
        let objects = &self.objects;
        self.indices
            .iter()
            .map(move |(i, val)| (Idx::from(i), &objects[*val]))
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.objects[..]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.objects[..]
    }

    #[inline]
    pub fn insert(&mut self, object: T) -> Idx
    where
        Idx: From<ContiguousArenaIndex>,
    {
        let i = self.objects.len();
        self.objects.push(object);
        let idx = self.indices.insert(i);
        self.rev_indices.push(idx);
        Idx::from(idx)
    }

    #[inline]
    pub fn remove(&mut self, handle: Idx) -> Option<T>
    where
        Idx: Into<ContiguousArenaIndex>,
    {
        let i = self.indices.remove(handle.into())?;
        let swapped_rev_id = self.rev_indices.last().cloned();
        let deleted_object = self.objects.swap_remove(i);
        let _ = self.rev_indices.swap_remove(i);

        if let Some(rev_id) = swapped_rev_id {
            self.indices[rev_id] = i;
        }

        Some(deleted_object)
    }
}

impl<Idx: Into<ContiguousArenaIndex>, T> std::ops::Index<Idx> for ContiguousArena<Idx, T> {
    type Output = T;

    #[inline]
    fn index(&self, i: Idx) -> &T {
        &self.objects[self.indices[i.into()]]
    }
}

impl<Idx: Into<ContiguousArenaIndex>, T> std::ops::IndexMut<Idx> for ContiguousArena<Idx, T> {
    #[inline]
    fn index_mut(&mut self, i: Idx) -> &mut T {
        &mut self.objects[self.indices[i.into()]]
    }
}

impl<Idx, T> AsRef<[T]> for ContiguousArena<Idx, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.objects
    }
}

impl<Idx, T> AsMut<[T]> for ContiguousArena<Idx, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.objects
    }
}
