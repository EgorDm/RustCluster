mod synthetic_2d_multithread;

use std::thread;
use crossbeam_channel::{bounded, Sender};

enum Test {
    TaskA(i32),
    TaskB(i32),
}

struct Orchestrator {
    tx: Sender<Test>,
}


fn main() {
    let (s, r) = bounded(0);

    let res = thread::spawn(move || {
        s.send(42).unwrap();
    });

    for num in r.iter().take(20) {
        println!("{}", num);
    }
}