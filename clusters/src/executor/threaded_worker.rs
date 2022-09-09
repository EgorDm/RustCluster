use std::thread;

pub struct ThreadedWorker {
    threads: Vec<thread::JoinHandle<()>>,
}

impl ThreadedWorker {

}