use remoc::{codec, prelude::*};
use std::{net::Ipv4Addr, sync::Arc, time::Duration};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use rand::rngs::StdRng;
use rand::SeedableRng;
use remoc::rtc::CallError;
use tokio::{net::TcpListener, sync::RwLock, time::sleep};

use clusters::executor::rtc::{Counter, IncreaseError, TCP_PORT, CounterServerSharedMut};
use clusters::state::GlobalState;
use clusters::local::{LocalActions, LocalState, ThinStats};
use clusters::params::options::ModelOptions;
use clusters::stats::{NIW, NIWStats};

/// Server object for the counting service, keeping the state.
pub struct CounterObj {
    local_states: Vec<LocalState<NIW>>,
    rng: StdRng,

    /// The current value.
    value: u32,
    /// The subscribed watchers.
    watchers: Vec<rch::watch::Sender<u32>>,
}

impl CounterObj {
    fn from_data(
        data: DMatrix<f64>,
        model_options: &ModelOptions<NIW>,
        n_clusters: usize,
        mut rng: StdRng,
    ) -> Self {
        // TODO: data hydration and initialization should probably be done separately since former depends on model options
        let mut local_states = vec![];
        local_states.push(
            LocalState::<NIW>::from_init(
                data,
                n_clusters, &model_options, &mut rng,
            )
        );
        Self {
            local_states,
            rng,
            value: 0,
            watchers: Vec::new(),
        }
    }
}

/// Implementation of remote counting service.
#[rtc::async_trait]
impl Counter for CounterObj {
    async fn collect_data_stats(&self) -> Result<NIWStats, CallError> {
        let data_stats = self.local_states.iter().map(LocalState::collect_data_stats).sum();
        Ok(data_stats)
    }

    async fn collect_stats(&self, n_clusters: usize) -> Result<ThinStats<NIW>, CallError> {
       Ok(
           self.local_states.iter()
            .map(|local_state| LocalState::collect_stats(local_state, 0..n_clusters))
            .sum()
       )
    }

    async fn update_sample_labels(&mut self, global_state: GlobalState<NIW>, is_final: bool) -> Result<(), CallError> {
        self.local_states.iter_mut().for_each(| local_state| {
            LocalState::apply_sample_labels_prim(&global_state, local_state, is_final, &mut self.rng);
            LocalState::apply_sample_labels_aux(&global_state, local_state, &mut self.rng);
        });
        Ok(())
    }

    async fn update_reset_clusters(&mut self, bad_clusters: Vec<usize>) -> Result<(), CallError> {
        self.local_states.iter_mut().for_each(|local_state| {
            LocalState::update_reset_clusters(local_state, &bad_clusters, &mut self.rng);
        });
        Ok(())
    }

    async fn update_remove_clusters(&mut self, removed_idx: Vec<usize>) -> Result<(), CallError> {
        self.local_states.iter_mut().for_each(|local_state| {
            LocalState::update_remove_clusters(local_state, &removed_idx);
        });
        Ok(())
    }

    async fn apply_split(&mut self, split_idx: Vec<(usize, usize)>) -> Result<(), CallError> {
        self.local_states.iter_mut().for_each(|local_state| {
            LocalActions::apply_split(local_state, &split_idx, &mut self.rng);
        });
        Ok(())
    }

    async fn apply_merge(&mut self, merge_idx: Vec<(usize, usize)>) -> Result<(), CallError> {
        self.local_states.iter_mut().for_each(|local_state| {
            LocalActions::apply_merge(local_state, &merge_idx);
        });
        Ok(())
    }


    async fn value(&self) -> Result<u32, rtc::CallError> {
        Ok(self.value)
    }

    async fn watch(&mut self) -> Result<rch::watch::Receiver<u32>, rtc::CallError> {
        // Create watch channel.
        let (tx, rx) = rch::watch::channel(self.value);
        // Keep the sender half in the watchers vector.
        self.watchers.push(tx);
        // And return the receiver half.
        Ok(rx)
    }

    async fn increase(&mut self, by: u32) -> Result<(), IncreaseError> {
        // Perform the addition if it does not overflow the counter.
        match self.value.checked_add(by) {
            Some(new_value) => self.value = new_value,
            None => return Err(IncreaseError::Overflow { current_value: self.value }),
        }

        // Notify all watchers and keep only the ones that are not disconnected.
        let value = self.value;
        self.watchers.retain(|watch| !watch.send(value).into_disconnected().unwrap());

        Ok(())
    }

    async fn count_to_value(
        &self, step: u32, delay: Duration,
    ) -> Result<rch::mpsc::Receiver<u32>, rtc::CallError> {
        // Create mpsc channel for counting.
        let (tx, rx) = rch::mpsc::channel(1);

        // Spawn a task to perform the counting.
        let value = self.value;
        tokio::spawn(async move {
            // Counting loop.
            for i in (0..value).step_by(step as usize) {
                // Send the value.
                if tx.send(i).await.into_disconnected().unwrap() {
                    // Abort the counting if the client dropped the
                    // receive half or disconnected.
                    break;
                }

                // Wait the specified delay.
                sleep(delay).await;
            }
        });

        // Return the receive half of the counting channel.
        Ok(rx)
    }
}

#[tokio::main]
async fn main() {
    let x_data: Array2<f64> = read_npy("examples/data/x.npy").unwrap();
    let x = DMatrix::from_row_slice(x_data.nrows(), x_data.ncols(), &x_data.as_slice().unwrap()).transpose();
    let y_data: Array1<i64> = read_npy("examples/data/y.npy").unwrap();
    let y = DVector::from_row_slice(&y_data.as_slice().unwrap());
    let y = y.map(|x| x as usize).into_owned().transpose();

    // Create a counter object that will be shared between all clients.
    // You could also create one counter object per connection.
    let dim = 2;
    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    model_options.outlier = None;

    let mut rng = StdRng::seed_from_u64(42);


    let counter_obj = Arc::new(RwLock::new(CounterObj::from_data(
        x, &model_options, 10, rng
    )));

    // Listen to TCP connections using Tokio.
    // In reality you would probably use TLS or WebSockets over HTTPS.
    println!("Listening on port {}. Press Ctrl+C to exit.", TCP_PORT);
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, TCP_PORT)).await.unwrap();

    loop {
        // Accept an incoming TCP connection.
        let (socket, addr) = listener.accept().await.unwrap();
        let (socket_rx, socket_tx) = socket.into_split();
        println!("Accepted connection from {}", addr);

        // Create a new shared reference to the counter object.
        let counter_obj = counter_obj.clone();

        // Spawn a task for each incoming connection.
        tokio::spawn(async move {
            // Create a server proxy and client for the accepted connection.
            //
            // The server proxy executes all incoming method calls on the shared counter_obj
            // with a request queue length of 1.
            //
            // Current limitations of the Rust compiler require that we explicitly
            // specify the codec.
            let (server, client) = CounterServerSharedMut::<_, codec::Default>::new(counter_obj, 1);

            // Establish a Remoc connection with default configuration over the TCP connection and
            // provide (i.e. send) the counter client to the client.
            remoc::Connect::io(remoc::Cfg::default(), socket_rx, socket_tx).provide(client).await.unwrap();

            // Serve incoming requests from the client on this task.
            // `true` indicates that requests are handled in parallel.
            server.serve(true).await;
        });
    }
}