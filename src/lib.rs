use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::parallel::SileroParallel;

mod parallel;
mod silero;
mod utils;

#[pyclass]
struct SileroVAD {
    model: SileroParallel,
    threshold: f32,
}

#[pymethods]
impl SileroVAD {
    #[new]
    #[pyo3(signature = (workers = 16, threshold = 0.5))]
    fn new(workers: usize, threshold: f32) -> PyResult<Self> {
        Ok(Self {
            model: SileroParallel::new(workers),
            threshold,
        })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        #[allow(unused)] kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // Get first argument (list of numpy arrays)
        let obj = args.get_item(0)?;
        let list_obj = obj.downcast::<PyList>().unwrap();

        let mut audios = Vec::with_capacity(list_obj.len());
        for item in list_obj.iter() {
            let arr = item.downcast::<PyArray1<f32>>()?;
            let data = unsafe { arr.as_array() };
            let audio = data.as_slice().unwrap().to_vec();
            audios.push(audio);
        }

        let results = self.model.detect_multi(audios, self.threshold);

        // Create a Python list from Rust Vec
        let py_list = PyList::new(py, results).unwrap();

        // Convert PyList -> Py<PyAny> (owning handle)
        Ok(py_list.into())
    }
}

#[pymodule]
fn silero_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SileroVAD>()?;
    Ok(())
}
