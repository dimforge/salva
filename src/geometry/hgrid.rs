use fnv::FnvHasher;

use std::collections::HashMap;

use crate::math::{Point, Real, Vector, DIM};

use std::hash::BuildHasher;

#[derive(Copy, Clone, Debug)]
pub struct DeterministicState;

impl BuildHasher for DeterministicState {
    type Hasher = FnvHasher;

    fn build_hasher(&self) -> FnvHasher {
        FnvHasher::with_key(1820)
    }
}

/// A grid based on spacial hashing.
#[derive(PartialEq, Debug, Clone)]
pub struct HGrid<T> {
    cells: HashMap<Point<i64>, Vec<T>, DeterministicState>,
    cell_width: Real,
}

impl<T> HGrid<T> {
    /// Initialize a grid where each cell has the width `cell_width`.
    pub fn new(cell_width: Real) -> Self {
        Self {
            cells: HashMap::with_hasher(DeterministicState),
            cell_width,
        }
    }

    /// The width of a cell of this spacial grid.
    pub fn cell_width(&self) -> Real {
        self.cell_width
    }

    fn quantify(value: Real, cell_width: Real) -> i64 {
        na::try_convert::<Real, f64>((value / cell_width).floor()).unwrap() as i64
    }

    fn quantify_ceil(value: Real, cell_width: Real) -> i64 {
        na::try_convert::<Real, f64>((value / cell_width).ceil()).unwrap() as i64
    }

    /// Computes the logical grid cell containing `point`.
    pub fn key(&self, point: &Point<Real>) -> Point<i64> {
        Point::from(point.coords.map(|e| Self::quantify(e, self.cell_width)))
    }

    /// Removes all elements from this grid.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Inserts the given `element` into the cell containing the given `point`.
    pub fn insert(&mut self, point: &Point<Real>, element: T) {
        let key = self.key(point);
        self.cells.entry(key).or_insert(Vec::new()).push(element)
    }

    /// Returns the element attached to the cell containing the given `point`.
    ///
    /// Returns `None` if the cell is empty.
    pub fn cell_containing_point(&self, point: &Point<Real>) -> Option<&Vec<T>> {
        let key = self.key(point);
        self.cells.get(&key)
    }

    /// An iterator through all the non-empty cells of this grid.
    ///
    /// The returned tuple include the cell indentifier, and the elements attached to this cell.
    pub fn cells(&self) -> impl Iterator<Item = (&Point<i64>, &Vec<T>)> {
        self.cells.iter()
    }

    /// The underlying hash map of this spacial grid.
    pub fn inner_table(&self) -> &HashMap<Point<i64>, Vec<T>, DeterministicState> {
        &self.cells
    }

    /// Get the content of the logical cell identified by `key`.
    pub fn cell(&self, key: &Point<i64>) -> Option<&Vec<T>> {
        self.cells.get(key)
    }

    /// An iterator through all the neighbors of the given cell.
    ///
    /// The given cell itself will be yielded by this iterator too.
    pub fn neighbor_cells(
        &self,
        cell: &Point<i64>,
        radius: Real,
    ) -> impl Iterator<Item = (Point<i64>, &Vec<T>)> {
        let cells = &self.cells;
        let quantified_radius = Self::quantify_ceil(radius, self.cell_width);

        CellRangeIterator::with_center(*cell, quantified_radius)
            .filter_map(move |cell| cells.get(&cell).map(|c| (cell, c)))
    }

    //    pub fn elements_close_to_point<'a>(
    //        &'a self,
    //        point: &Point<Real>,
    //        radius: Real,
    //    ) -> impl Iterator<Item = &T>
    //    {
    //        let key = self.key(point, self.cell_width);
    //        // FIXME: we could sometimes avoid the `+ 1` by taking into account the point location on the cell.
    //        let quantified_radius = Self::quantify(radius, self.cell_width) + 1;
    //        let cells = &self.cells;
    //
    //        CellRangeIterator::with_center(key, quantified_radius)
    //            .flat_map(move |cell| cells.get(&cell).into_iter())
    //            .flat_map(|cells| cells.iter())
    //    }

    /// An iterator through all the cells intersecting the given AABB.
    pub fn cells_intersecting_aabb(
        &self,
        mins: &Point<Real>,
        maxs: &Point<Real>,
    ) -> impl Iterator<Item = (Point<i64>, &Vec<T>)> {
        let cells = &self.cells;
        let start = self.key(mins);
        let end = self.key(maxs);

        CellRangeIterator::new(start, end)
            .filter_map(move |cell| cells.get(&cell).map(|c| (cell, c)))
    }

    //    pub fn elements_containing_point(&self, point: &Point<Real>) -> impl Iterator<Item = &T> {
    //        std::iter::empty()
    //    }
}

struct CellRangeIterator {
    start: Point<i64>,
    end: Point<i64>,
    curr: Point<i64>,
    done: bool,
}

impl CellRangeIterator {
    fn new(start: Point<i64>, end: Point<i64>) -> Self {
        Self {
            start,
            end,
            curr: start,
            done: false,
        }
    }

    fn with_center(center: Point<i64>, radius: i64) -> Self {
        let start = center - Vector::repeat(radius as i64);
        Self {
            start,
            end: center + Vector::repeat(radius as i64),
            curr: start,
            done: false,
        }
    }
}

impl Iterator for CellRangeIterator {
    type Item = Point<i64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.curr == self.end {
            self.done = true;
            Some(self.curr)
        } else {
            let result = self.curr;

            for i in 0..DIM {
                self.curr[i] += 1;

                if self.curr[i] > self.end[i] {
                    self.curr[i] = self.start[i];
                } else {
                    break;
                }
            }

            Some(result)
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    #[cfg(feature = "dim2")]
    fn grid_neighbor_iterator() {
        use super::CellRangeIterator;
        use crate::math::Point;

        let expected = [
            Point::new(-1, 0),
            Point::new(0, 0),
            Point::new(1, 0),
            Point::new(2, 0),
            Point::new(3, 0),
            Point::new(-1, 1),
            Point::new(0, 1),
            Point::new(1, 1),
            Point::new(2, 1),
            Point::new(3, 1),
            Point::new(-1, 2),
            Point::new(0, 2),
            Point::new(1, 2),
            Point::new(2, 2),
            Point::new(3, 2),
            Point::new(-1, 3),
            Point::new(0, 3),
            Point::new(1, 3),
            Point::new(2, 3),
            Point::new(3, 3),
        ];

        let iter = CellRangeIterator::with_center(Point::new(1, 2), 2);

        assert!(iter.zip(expected.iter()).all(|(a, b)| a == *b))
    }
}
