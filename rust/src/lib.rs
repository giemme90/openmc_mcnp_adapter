use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyInt;
use pyo3::types::PyString;
use rayon::prelude::*;
use rayon::iter::ParallelBridge;

fn position_value_max_absolute(v: &Vec<f64>)-> (usize, f64) {
    v.iter()
            .enumerate()
            .fold((0, v[0]), |(idx_max, val_max), (idx, val)| {
                if val_max.abs() > val.abs() {
                    (idx_max, val_max)
                } else {
                    (idx, *val)
                }
            })
}

#[derive(Debug,PartialEq)]
enum SameDifferentOpposite {
    Same=1,
    Different=0,
    Opposite=-1
}

struct Surface {
    id: i64,
    kind: String,
    coefficients: Vec<f64>
}

struct Plane {
    id: i64,
    kind: String,
    coefficients: Vec<f64>
}

trait ApproximateEq {
    const EPSILON: f64 = 1e-12;

    fn approximate_eq(&self, other: &Self) -> SameDifferentOpposite; // see here
}

impl ApproximateEq for f64 {
    fn approximate_eq(&self, other: &f64) -> SameDifferentOpposite{
       if (self - other).abs() <= <f64 as ApproximateEq>::EPSILON {
           return SameDifferentOpposite::Same
       } else if (self + other).abs() <= <f64 as ApproximateEq>::EPSILON {
           return SameDifferentOpposite::Opposite
       } else {
           return SameDifferentOpposite::Different
       } 
    }
}

trait ApproximateEqDynamic {
    const EPSILON: f64 = 1e-12;

    fn approximate_eq_dynamic(&self, other: &Self) -> SameDifferentOpposite; 
}

impl ApproximateEqDynamic for f64 {
    fn approximate_eq_dynamic(&self, other: &f64) -> SameDifferentOpposite{
       if (self - other).abs() <= (self + other).abs()*<f64 as ApproximateEqDynamic>::EPSILON {
           return SameDifferentOpposite::Same
       } else if (self + other).abs() <= (self - other).abs()*<f64 as ApproximateEqDynamic>::EPSILON {
           return SameDifferentOpposite::Opposite
       } else {
           return SameDifferentOpposite::Different
       } 
    }
}

impl ApproximateEqDynamic for Plane {
    fn approximate_eq_dynamic(&self, other: &Self) -> SameDifferentOpposite {
        if self.coefficients.len() != other.coefficients.len() || self.kind != other.kind {
           return SameDifferentOpposite::Different
        }
        let (max_idx, max_val) = position_value_max_absolute(&self.coefficients);
        let ratio = (other.coefficients[max_idx] / max_val).abs();
        
        let mut compare_vec = self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .filter(|(&vs, &v0)| (vs, v0) != (0.0, 0.0))
            .map(|(vs, vo)| (ratio*vs).approximate_eq_dynamic(&vo) );

        let first = compare_vec.next().unwrap();
        
        compare_vec.all(|elem| elem == first).then(|| first).unwrap_or_else(|| SameDifferentOpposite::Different)
        
                
    }
}

impl ApproximateEqDynamic for Surface {
    fn approximate_eq_dynamic(&self, other: &Self) -> SameDifferentOpposite {
        if self.coefficients.len() != other.coefficients.len() || self.kind != other.kind {
           return SameDifferentOpposite::Different
        }

        let mut compare_vec = self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .filter(|(&vs, &v0)| (vs, v0) != (0.0, 0.0))
            .map(|(vs, vo)| vs.approximate_eq_dynamic(&vo) );

        let first = compare_vec.next().unwrap();
        
        compare_vec.all(|elem| elem == first).then(|| first).unwrap_or_else(|| SameDifferentOpposite::Different)
        
                
    }
}

impl ApproximateEq for Surface {
    fn approximate_eq(&self, other: &Self) -> SameDifferentOpposite {
        if self.coefficients.len() != other.coefficients.len() || self.kind != other.kind {
           return SameDifferentOpposite::Different
        }

        let mut compare_vec = self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .filter(|(&vs, &v0)| (vs, v0) != (0.0, 0.0))
            .map(|(vs, vo)| vs.approximate_eq(&vo) );

        let first = compare_vec.next().unwrap();
        compare_vec.all(|elem| elem == first).then(|| first).unwrap_or_else(|| SameDifferentOpposite::Different)
        
                
    }
}

impl ApproximateEq for Plane {
    fn approximate_eq(&self, other: &Self) -> SameDifferentOpposite {
        if self.coefficients.len() != other.coefficients.len() || self.kind != other.kind {
           return SameDifferentOpposite::Different
        }

        let mut compare_vec = self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .filter(|(&vs, &v0)| (vs, v0) != (0.0, 0.0))
            .map(|(vs, vo)| vs.approximate_eq(&vo) );

        let first = compare_vec.next().unwrap();
        compare_vec.all(|elem| elem == first).then(|| first).unwrap_or_else(|| SameDifferentOpposite::Different)
        
                
    }
}



#[pyfunction]
fn compare(surfaces: &PyDict, tipo: String) -> PyResult<std::collections::HashMap<i64,i64>> {
    
    let surfaces_keys = surfaces.keys();
    let mut surfaces_hashmap: std::collections::HashMap<i64, Surface> = std::collections::HashMap::new();
    let mut planes_hashmap: std::collections::HashMap<i64, Plane> = std::collections::HashMap::new();
    for pyid0 in surfaces_keys.iter(){
        let id0 = pyid0.downcast::<PyInt>()?.extract::<i64>()?;
        let s0 = surfaces.get_item(&id0).unwrap().downcast::<PyDict>()?;
        let s0_kind = s0.get_item(String::from("kind")).unwrap().downcast::<PyString>()?.extract::<String>()?;
        let s0_coefficients = s0.get_item(String::from("coefficients")).unwrap().downcast::<PyList>()?.extract::<Vec<f64>>()?;
        if s0_kind == "plane".to_string() {
            let plane1 = Plane{id:id0, kind:s0_kind, coefficients:s0_coefficients};
            planes_hashmap.insert(id0, plane1);
        }else{
            let surf1 = Surface{id:id0, kind:s0_kind, coefficients:s0_coefficients};
            surfaces_hashmap.insert(id0, surf1);
        }
    }
    
    

    let planes_keys: Vec<&i64> = planes_hashmap.keys().collect();
    let combinations_planes = planes_keys.iter()
           .enumerate()
           .flat_map (|(i, a)| planes_keys[i+1..].iter().map (move |b| (a, b)));

    //let res: Vec<(i64, i64)> = planes_hashmap.keys().flat_map(|i| planes_hashmap.keys().map(move |j| (i, j))).filter(|&x| x.1>x.0).collect();
    let par_iter_planes = combinations_planes.par_bridge().map(|(id0, id1)| {
       // for (id0, id1) in combinations_planes {
            let plane1 = planes_hashmap.get(&id0).unwrap();
            let plane2 = planes_hashmap.get(&id1).unwrap();
            let result_comparison;
            if tipo == String::from("Dynamic"){
                result_comparison = plane1.approximate_eq_dynamic(&plane2);
            } else {
                result_comparison = plane1.approximate_eq(&plane2);
            }
            if result_comparison != SameDifferentOpposite::Different {
                let value = **id0 * result_comparison as i64;
                (**id1, value)
            } else {
                (0, 0)
            }
        }
    
     ).filter(|(id0, id1)| (*id0, *id1) != (0, 0));

    
    let mut same_opposite: std::collections::HashMap<i64, i64> = par_iter_planes.map(|(key, value)| (key, value)).collect();
 
    let surfs_keys: Vec<&i64>= surfaces_hashmap.keys().collect();
    let combinations_surfs = surfs_keys.iter()
           .enumerate()
           .flat_map (|(i, a)| surfs_keys [i+1..].iter().map (move |b| (a, b)));
    for (id0, id1) in combinations_surfs {
        let surf1 = surfaces_hashmap.get(&id0).unwrap();
        let surf2 = surfaces_hashmap.get(&id1).unwrap();
        let result_comparison;
        if tipo == String::from("Dynamic"){
            result_comparison = surf1.approximate_eq_dynamic(&surf2);
        } else {
            result_comparison = surf1.approximate_eq(&surf2);
        }
        if result_comparison != SameDifferentOpposite::Different {
            let value = *id0 * result_comparison as i64;
//
            same_opposite.insert(**id1, value);
            //same_opposite.insert(**id0, value);
        }
    }
    Ok(same_opposite)
}

/// A Python module implemented in Rust.
#[pymodule]
fn surfaces_comparison(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    Ok(())
}
