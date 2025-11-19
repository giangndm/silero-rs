use anyhow::Result;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, Value};
use pyo3::pyclass;

use crate::utils::get_hub_model_file;

const CHUNK_SIZE: usize = 512;
const PADDING_SIZE: usize = 64;

#[pyclass]
pub struct Segment {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
}

pub struct SileroVadOrtSession {
    context: Vec<f32>,
    state: Value,
}

pub struct SileroVadOrt {
    model: Session,
}

impl SileroVadOrt {
    pub fn from_pretrained(repo_id: &str, file_name: &str) -> Result<Self> {
        let path = get_hub_model_file(repo_id, None, file_name)?;
        let model_builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;
        let model = model_builder.commit_from_file(path)?;

        Ok(Self { model })
    }

    pub fn detect_long(&mut self, audio: &[f32], threshold: f32) -> Result<Vec<Segment>> {
        let mut session = self.new_session()?;
        let mut segments = Vec::new();
        let mut start = None;
        for (index, chunk) in audio.chunks(CHUNK_SIZE).enumerate() {
            if chunk.len() < CHUNK_SIZE {
                break;
            }
            let prob = self.detect(&mut session, chunk)?;
            if prob > threshold {
                if start.is_none() {
                    start = Some(index * CHUNK_SIZE);
                }
            } else {
                if let Some(start) = start.take() {
                    segments.push(Segment {
                        start,
                        end: (index + 1) * CHUNK_SIZE,
                    });
                }
            }
        }
        if let Some(start) = start.take() {
            segments.push(Segment {
                start,
                end: audio.len(),
            });
        }
        Ok(segments)
    }

    pub fn new_session(&self) -> Result<SileroVadOrtSession> {
        Ok(SileroVadOrtSession {
            context: vec![0.0f32; PADDING_SIZE + CHUNK_SIZE],
            state: Tensor::from_array(([2usize, 1, 128], vec![0.0f32; 256]))?.into(),
        })
    }

    pub fn detect(&mut self, session: &mut SileroVadOrtSession, audio: &[f32]) -> Result<f32> {
        assert_eq!(audio.len(), CHUNK_SIZE, "Audio length must be {CHUNK_SIZE}");
        session.context[PADDING_SIZE..].copy_from_slice(&audio);
        let mut outputs = self.model.run(ort::inputs![
            "input" => Tensor::from_array(([1, CHUNK_SIZE + PADDING_SIZE], session.context.clone()))?,
            "state" => &session.state,
            "sr" => Tensor::from_array(([1], vec![16000i64]))?,
        ])?;
        let (_output_dim, output) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = output[0];
        let state = outputs
            .remove("stateN")
            .ok_or_else(|| anyhow::anyhow!("'stateN' not found in model outputs"))?;
        session.state = state;
        // move last PADDING_SIZE samples to left
        session.context.copy_within(CHUNK_SIZE.., 0);
        Ok(prob)
    }
}
