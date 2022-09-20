use pyo3::prelude::*;
use numpy::nalgebra::{U1, Dynamic};
use numpy::{Ix2, PyArray, PyReadonlyArrayDyn, ToPyArray};
use mixturs::{NIW};

#[macro_export]
macro_rules! pyacessors {
    {
        impl $class:ident {
            $(
                get_set($field:ident, $setter:ident, $ty:ty)
            )*
        }
    } => {
        #[pymethods]
        impl $class {
            $(
                #[getter]
                fn $field(&self) -> $ty {
                    self.inner.$field
                }

                #[setter]
                fn $setter(&mut self, value: $ty) {
                    self.inner.$field = value;
                }
            )*
        }
    };
}

#[pyclass]
pub struct Model {
    inner: mixturs::Model<NIW>,
}

#[pymethods]
impl Model {
    #[new]
    pub fn new(mut model_options: ModelOptions) -> Self {
        model_options.inner.outlier = None;
        Self {
            inner: mixturs::Model::from_options(model_options.inner),
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.inner.n_clusters()
    }

    pub fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    pub fn fit(
        &mut self,
        x: PyReadonlyArrayDyn<f64>,
        fit_options: &FitOptions,
        y: Option<PyReadonlyArrayDyn<usize>>,
    ) {
        let x = x.try_as_matrix::<Dynamic, Dynamic, Dynamic, Dynamic>().unwrap().clone_owned();
        let y = y.map(|y| y.try_as_matrix::<Dynamic, U1, U1, Dynamic>().unwrap().transpose());

        let mut callback = mixturs::MonitoringCallback::from_data(
            mixturs::callback::EvalData::from_sample(&x, y.as_ref(), fit_options.eval_points)
        );
        callback.set_verbose(fit_options.verbose);
        if fit_options.nmi {
            callback.add_metric(mixturs::NMI);
        }
        if fit_options.aic {
            callback.add_metric(mixturs::AIC);
        }
        if fit_options.bic {
            callback.add_metric(mixturs::BIC);
        }

        self.inner.fit(
            x,
            &fit_options.inner,
            Some(callback),
        );
    }

    pub fn predict<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<(
        &'py PyArray<f64, Ix2>,
        &'py PyArray<usize, Ix2>,
    )> {
        let data = data.try_as_matrix::<Dynamic, Dynamic, Dynamic, Dynamic>().unwrap().clone_owned();

        let (probs, labels) = self.inner.predict(data);

        Ok((
            probs.to_pyarray(py),
            labels.to_pyarray(py),
        ))
    }

    pub fn cluster_params(&self) -> Vec<MultivariateNormal> {
        self.inner.params().clusters.iter().map(|c|
            MultivariateNormal { inner: c.prim.dist.clone() }
        ).collect()
    }

    pub fn cluster_weights(&self) -> Vec<f64> {
        self.inner.params().weights.clone()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct FitOptions {
    inner: mixturs::FitOptions,

    #[pyo3(get, set)]
    verbose: bool,
    #[pyo3(get, set)]
    nmi: bool,
    #[pyo3(get, set)]
    aic: bool,
    #[pyo3(get, set)]
    bic: bool,
    #[pyo3(get, set)]
    eval_points: usize,
}

#[pymethods]
impl FitOptions {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: mixturs::FitOptions::default(),

            verbose: true,
            nmi: false,
            aic: false,
            bic: false,
            eval_points: 1000,
        }
    }
}

pyacessors! {
    impl FitOptions {
        get_set(seed, set_seed, u64)
        get_set(reuse, set_reuse, bool)
        get_set(init_clusters, set_init_clusters, usize)
        get_set(max_clusters, set_max_clusters, usize)
        get_set(iters, set_iters, usize)
        get_set(argmax_sample_stop, set_argmax_sample_stop, usize)
        get_set(iter_split_stop, set_iter_split_stop, usize)
        get_set(workers, set_workers, i32)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ModelOptions {
    inner: mixturs::ModelOptions<NIW>,
}

#[pymethods]
impl ModelOptions {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self {
            inner: mixturs::ModelOptions::default(dim),
        }
    }

    pub fn data_dist(&self) -> NIWParams {
        NIWParams { inner: self.inner.data_dist.clone() }
    }

    pub fn set_data_dist(&mut self, data_dist: NIWParams) {
        self.inner.data_dist = data_dist.inner;
    }

    pub fn outlier_removal(&self) -> Option<OutlierRemoval> {
        self.inner.outlier.as_ref().map(|o| OutlierRemoval { inner: o.clone() })
    }

    pub fn set_outlier_removal(&mut self, outlier_removal: Option<OutlierRemoval>) {
        self.inner.outlier = outlier_removal.map(|o| o.inner);
    }
}

pyacessors! {
    impl ModelOptions {
        get_set(alpha, set_alpha, f64)
        get_set(dim, set_dim, usize)
        get_set(burnout_period, set_burnout_period, usize)
        get_set(hard_assignment, set_hard_assignment, bool)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct NIWParams {
    inner: mixturs::stats::NIWParams,
}

#[pymethods]
impl NIWParams {
    #[new]
    pub fn new(
        kappa: f64,
        mu: PyReadonlyArrayDyn<f64>,
        nu: f64,
        psi: PyReadonlyArrayDyn<f64>,
    ) -> Self {
        let mu = mu.try_as_matrix::<Dynamic, U1, Dynamic, U1>().unwrap().clone_owned();
        let psi = psi.try_as_matrix::<Dynamic, Dynamic, Dynamic, Dynamic>().unwrap().clone_owned();

        Self {
            inner: mixturs::stats::NIWParams::new(kappa, mu, nu, psi),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct OutlierRemoval {
    inner: mixturs::params::OutlierRemoval<NIW>,
}

#[pymethods]
impl OutlierRemoval {
    #[new]
    pub fn new(
        weight: f64,
        dist: NIWParams,
    ) -> Self {
        Self {
            inner: mixturs::params::OutlierRemoval { weight, dist: dist.inner },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MultivariateNormal {
    inner: mixturs::stats::MultivariateNormal,
}

#[pymethods]
impl MultivariateNormal {
    pub fn mu<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix2> {
        self.inner.mu().to_pyarray(py)
    }

    pub fn cov<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix2> {
        self.inner.cov().to_pyarray(py)
    }
}

#[pymodule]
fn mixtupy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<FitOptions>()?;
    m.add_class::<ModelOptions>()?;
    m.add_class::<NIWParams>()?;
    m.add_class::<Model>()?;
    m.add_class::<MultivariateNormal>()?;
    m.add_class::<OutlierRemoval>()?;
    Ok(())
}
