use flume::{unbounded, Receiver, Sender};

use crate::silero::{Segment, SileroVadOrt};

type AudioData = (usize, f32, Vec<f32>);
type SegmentsData = (usize, Vec<Segment>);

pub struct SileroParallel {
    workers_tx: Vec<Sender<AudioData>>,
    response_rx: Receiver<SegmentsData>,
}

impl SileroParallel {
    pub fn new(workers: usize) -> Self {
        let (response_tx, response_rx) = unbounded::<SegmentsData>();
        let mut workers_tx = Vec::with_capacity(workers);
        for _ in 0..workers {
            let response_tx = response_tx.clone();
            let (worker_tx, worker_rx) = unbounded::<AudioData>();
            std::thread::spawn(move || {
                let mut silero =
                    SileroVadOrt::from_pretrained("Narsil/silero", "silero_vad_16k_op15.onnx")
                        .unwrap();
                while let Ok((index, threshold, audio)) = worker_rx.recv() {
                    // println!("Worker got {} {} samples", index, audio.len());
                    let result = silero.detect_long(&audio, threshold).unwrap();
                    response_tx.send((index, result)).unwrap();
                }
            });
            workers_tx.push(worker_tx);
        }
        Self {
            workers_tx,
            response_rx,
        }
    }

    pub fn detect_multi(&self, audios: Vec<Vec<f32>>, threshold: f32) -> Vec<Vec<Segment>> {
        let mut results = Vec::with_capacity(audios.len());
        for _ in 0..audios.len() {
            results.push(vec![]);
        }

        for (index, audio) in audios.into_iter().enumerate() {
            // println!("Main send {} {} samples", index, audio.len());
            self.workers_tx[index % self.workers_tx.len()]
                .send((index, threshold, audio))
                .unwrap();
        }
        // println!("Main send done");

        // println!("Main wait receive");
        let mut count = 0;
        while let Ok((index, result)) = self.response_rx.recv() {
            results[index] = result;
            count += 1;
            // println!("Count: {}/{}", count, results.len());
            if count == results.len() {
                break;
            }
        }
        // println!("Main receive done");

        results
    }
}
