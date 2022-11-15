<!-- markdownlint-disable -->
<div id="top"></div>
<div align="center">
    <h1>mixturs</h1>
    <p>
        <a href="https://crates.io/crates/mixturs">
            <img alt="Crates.io" src="https://img.shields.io/crates/v/mixturs">
        </a>
        <a href="LICENSE">
            <img src="https://img.shields.io/github/license/EgorDm/mixturs" alt="License">
        </a>
        <a href="https://docs.rs/mixturs">
            <img src="https://img.shields.io/docsrs/mixturs" alt="Docs">
        </a>
        <a href="https://app.codecov.io/github/EgorDm/mixturs">
            <img src="https://img.shields.io/codecov/c/github/EgorDm/mixturs" alt="Coverage">
        </a>
    </p>
    <p>
        <b>Unofficial implementation of Dirichlet Process Mixture Model Sub-Clusters algorithm.</b>
    </p>
</div>
<p align="center">
  <a href="#features">Features</a> •
  <a href="#usage">Usage</a> •
  <a href="#examples">Examples</a>
</p>
<!-- markdownlint-enable -->


![Image Demo](docs/resources/image_clustering.png)


To use as a library, add the following to your Cargo.toml. Executable builds can be found at https://github.com/EgorDm/mixturs/releases.

```toml
[dependencies]
mixturs = "0.1"
```

## Features

* Cluster points without knowing the number of clusters in advance
* Fastest CPU implementation of the algorithm
* Python bindings to cluster numpy data
* Command line tool for generating segmented images from JPG/PNG input files

## Examples

### [Rust Examples](https://github.com/EgorDm/mixturs/tree/master/mixturs/examples):

```rust
// Load data into a col major matrix
let x: DMatrix<f64> = ...;

// Set model options
let mut model_options = ModelOptions::<NIW>::default(dim);
model_options.alpha = 100.0;
model_options.outlier = None;

// Set fit options
let mut fit_options = FitOptions::default();
fit_options.init_clusters = 1;

// Configure callbacks
let mut callback = MonitoringCallback::from_data(
    EvalData::from_sample(&x, Some(&y), 1000)
);
callback.add_metric(AIC);
callback.add_callback(PlotCallback::new(
    3,
    "examples/data/plot/synthetic_2d".into(),
    EvalData::from_sample(&x, None, 1000)
));
callback.set_verbose(true);

// Fit the model
let mut model = Model::from_options(model_options);
model.fit(
    x.clone_owned(),
    &fit_options,
    Some(callback),
);
```

### [Python Examples](https://github.com/EgorDm/mixturs/tree/master/mixturs-python/examples):

```python
import numpy as np
from mixtupy import *

# Load data
x = ...

# Configure model
mo = ModelOptions(2)
model = Model(mo)

# Fit model
fo = FitOptions()
fo.iters = 200
fo.aic = True
model.fit(x, fo, y=y)

# Predict data point labels
probs, labels = model.predict(x)

# Extract cluster parameters
print(model.cluster_weights())
print(model.cluster_params())
```

## Reference

[1] *J. Chang and J. W. Fisher III, “Parallel Sampling of DP Mixture Models using Sub-Cluster Splits,” in Advances in Neural Information Processing Systems, 2013.*

[2] *O. Dinari, A. Yu, O. Freifeld, and J. Fisher, “Distributed MCMC Inference in Dirichlet Process Mixture Models Using Julia,” in 2019 19th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGRID).*
