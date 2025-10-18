use anyhow::Result;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, Value};
use pyo3::pyclass;

use crate::utils::get_hub_model_file;

#[pyclass]
pub struct Segment {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
}

pub struct SileroVadOrtSession {
    state: Value,
}

pub struct SileroVadOrt {
    model: Session,
    chunk_size: usize,
}

impl SileroVadOrt {
    pub fn from_pretrained(repo_id: &str, file_name: &str, chunk_size: usize) -> Result<Self> {
        let path = get_hub_model_file(repo_id, None, file_name)?;
        let model_builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;
        let model = model_builder.commit_from_file(path)?;

        Ok(Self { model, chunk_size })
    }

    pub fn detect_long(&mut self, audio: &[f32], threshold: f32) -> Result<Vec<Segment>> {
        let mut session = self.new_session()?;
        let mut segments = Vec::new();
        let mut start = None;
        for (index, chunk) in audio.chunks(self.chunk_size).enumerate() {
            if chunk.len() < self.chunk_size {
                break;
            }
            let prob = self.detect(&mut session, chunk)?;
            if prob > threshold {
                if start.is_none() {
                    start = Some(index * self.chunk_size);
                }
            } else {
                if let Some(start) = start.take() {
                    segments.push(Segment {
                        start,
                        end: (index + 1) * self.chunk_size,
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
            state: Tensor::from_array(([2usize, 1, 128], vec![0.0f32; 256]))?.into(),
        })
    }

    pub fn detect(&mut self, session: &mut SileroVadOrtSession, audio: &[f32]) -> Result<f32> {
        assert_eq!(
            audio.len(),
            self.chunk_size,
            "Audio length must be {}",
            self.chunk_size
        );
        let mut outputs = self.model.run(ort::inputs![
            "input" => Tensor::from_array(([1, self.chunk_size], audio.to_vec()))?,
            "state" => &session.state,
            "sr" => Tensor::from_array(([1], vec![16000i64]))?,
        ])?;
        let (_output_dim, output) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = output[0];
        let state = outputs
            .remove("stateN")
            .ok_or_else(|| anyhow::anyhow!("'stateN' not found in model outputs"))?;
        session.state = state;
        Ok(prob)
    }
}
