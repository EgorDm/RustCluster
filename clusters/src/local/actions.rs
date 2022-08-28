use std::marker::PhantomData;
use rand::Rng;
use crate::local::state::LocalState;
use crate::stats::GaussianPrior;

pub struct LocalActions<P: GaussianPrior> {
    _phantoms: PhantomData<P>
}

impl<P: GaussianPrior> LocalActions<P> {
    pub fn apply_split<R: Rng>(
        state: &mut LocalState<P>,
        split_idx: &[(usize, usize)],
        rng: &mut R,
    ) {
        for (ki, kj) in split_idx.iter().cloned() {
            for (label, label_aux) in state.labels.iter_mut().zip(state.labels_aux.iter_mut()) {
                if *label == ki {
                    *label = if *label_aux == 0 { ki } else { kj };
                    *label_aux = rng.gen_range(0..2);
                }
            }
        }
    }

    pub fn apply_merge(
        state: &mut LocalState<P>,
        merge_idx: &[(usize, usize)],
    ) {
        for (i, j) in merge_idx.iter().cloned() {
            for (label, label_aux) in state.labels.iter_mut().zip(state.labels_aux.iter_mut()) {
                if *label == i {
                    *label_aux = 0;
                }
                if *label == j {
                    *label = i;
                    *label_aux = 2;
                }
            }
        }

    }
}
