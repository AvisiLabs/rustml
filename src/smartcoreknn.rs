extern crate openblas_src;
extern crate postgres;


use std::fs::File;
use std::io::Write;
use std::ops::Div;

use ndarray::{arr1, Array, Array1, Array2, Axis};
use num_traits::{ToPrimitive, Zero};
use plotters::prelude::*;
use postgres::{Client, NoTls};
use rand;
use rand::Rng;
use smartcore::linalg::BaseMatrix;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::math::num::RealNumber;
// Model performance
use smartcore::metrics::{mean_squared_error};
use smartcore::model_selection::train_test_split;
// Imports for KNN classifier
use smartcore::neighbors::knn_classifier::KNNClassifier;

struct Track {
    danceability: f64,
    energy: f64,
    key: i64,
    loudness: f64,
    mode: i64,
    speechiness: f64,
    instrumentalness: f64,
    liveness: f64,
    valence: f64,
    tempo: f64,
    time_signature: i64,
}

trait ToArray {
    fn to_array(&self) -> Array1<f64>;
}

impl ToArray for Track {
    fn to_array(&self) -> Array1<f64> {
        let vector = Vec::from([self.danceability,
            self.energy,
            self.key.to_f64().unwrap(),
            self.loudness,
            self.mode.to_f64().unwrap(),
            self.speechiness,
            self.instrumentalness,
            self.liveness,
            self.valence,
            self.tempo,
            self.time_signature.to_f64().unwrap()
        ]);
        return Array::from(arr1(&*vector));
    }
}

fn main() {
    let mut dataset = get_tracks("albert").expect("can't get dataset from database.");
    let y = fill_dataset_return_y(&mut dataset);
    let records = Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let knn = KNNClassifier::fit(&x_train, &y_train, Default::default()).unwrap();

    let predictions = knn.predict(&x_test).unwrap();
    println!("MSE: {}", mean_squared_error(&y_test, &predictions));
    {
        let knn_bytes = bincode::serialize(&knn).expect("Can not serialize the model");
        File::create("knn.model")
            .and_then(|mut f| f.write_all(&knn_bytes))
            .expect("Can not persist model");
    }
    scatterplot(
        &x,
        Some(&predictions.into_iter().map(|f| f as usize).collect()),
        "test",
    )
        .unwrap();
}

pub fn get_dataset(table: &str) -> (Array2<f64>, Vec<f64>) {
    let data_array = get_tracks(table).expect("can't get dataset for some table");
    let targets: Vec<f64> = vec![4.0; data_array.nrows()];
    return (data_array, targets);
}

pub fn get_dataset_kevin() -> (Array2<f64>, Vec<f64>) {
    let mut data_array = get_tracks("kevin").expect("can't get dataset for kevin");
    let mut targets: Vec<f64> = vec![0.0; data_array.nrows()];
    let mut i = 0;
    for rowerik in get_tracks("erik").expect("can't get dataset for erik").rows() {
        if (i < 54) {
            i += 1;
            data_array.push_row(rowerik).unwrap();
        }
    }
    targets.append(&mut vec![1.0; data_array.nrows() - targets.len()]);
    return (data_array, targets);
}

pub fn fill_dataset_return_y(mut array: &mut Array2<f64>) -> Vec<f64> {
    let mut targets: Vec<f64> = vec![0.0; array.nrows()];
    for roweline in get_tracks("eline").expect("boom").rows() {
        array.push_row(roweline).unwrap();
    }
    let mut ewelements = vec![1.0; array.nrows()];
    targets.append(&mut ewelements);
    for rowerik in get_tracks("erik").expect("boom").rows() {
        array.push_row(rowerik).unwrap();
    }
    targets.append(&mut vec![2.0; array.nrows() - targets.len()]);
    for rowesra in get_tracks("esra").expect("boom").rows() {
        array.push_row(rowesra).unwrap();
    }
    targets.append(&mut vec![3.0; array.nrows() - targets.len()]);
    for rowjordi in get_tracks("jordi").expect("boom").rows() {
        array.push_row(rowjordi).unwrap();
    }
    targets.append(&mut vec![4.0; array.nrows() - targets.len()]);
    for rowkevin in get_tracks("kevin").expect("boom").rows() {
        array.push_row(rowkevin).unwrap();
    }
    targets.append(&mut vec![5.0; array.nrows() - targets.len()]);
    for rowmarcel in get_tracks("marcel").expect("boom").rows() {
        array.push_row(rowmarcel).unwrap();
    }
    targets.append(&mut vec![6.0; array.nrows() - targets.len()]);
    targets
}

pub fn get_tracks(table: &str) -> Result<Array2<f64>, postgres::Error> {
    let mut conn = Client::connect("postgresql://admin:fkdvaQmAK82a4cGfdZSQ8rH6cyE2gCac@localhost/rustspotify", NoTls)
        .unwrap();
    let mut vec = Vec::new();
    let query = format!("SELECT danceability, energy, key, loudness, mode, speechiness, instrumentalness, liveness, valence, tempo, time_signature FROM {table}");
    let mut i = 0;
    for row in conn.query(query.as_str(), &[])? {
        let track = Track {
            danceability: row.get(0),
            energy: row.get(1),
            key: row.get(2),
            loudness: row.get(3),
            mode: row.get(4),
            speechiness: row.get(5),
            instrumentalness: row.get(6),
            liveness: row.get(7),
            valence: row.get(8),
            tempo: row.get(9),
            time_signature: row.get(10),
        };
        let track_array: Array1<f64> = track.to_array();
        vec.insert(i as usize, track_array);
        i += 1;
    }
    let array = to_array2(vec.as_slice()).unwrap();
    return Result::Ok(array);
}

fn to_array2<T: Copy>(source: &[Array1<T>]) -> Result<Array2<T>, impl std::error::Error> {
    let width = source.len();
    let flattened: Array1<T> = source.into_iter().flat_map(|row| row.to_vec()).collect();
    let height = flattened.len() / width;
    flattened.into_shape((width, height))
}


/// Draw a scatterplot of `data` with labels `labels`
/// We use Plotters library to draw scatter plot.
/// https://docs.rs/plotters/0.3.0/plotters/
pub fn scatterplot(
    data: &DenseMatrix<f64>,
    labels: Option<&Vec<usize>>,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();

    let x_min = (min(data, 0) - 1.0) as f64;
    let x_max = (max(data, 0) + 1.0) as f64;
    let y_min = (min(data, 1) - 1.0) as f64;
    let y_max = (max(data, 1) + 1.0) as f64;

    root.fill(&WHITE)?;
    let root = root.margin(15, 15, 15, 15);

    let data_values: Vec<f64> = data.iter().map(|v| v as f64).collect();

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .draw()?;
    match labels {
        Some(labels) => {
            scatter_ctx.draw_series(data_values.chunks(2).zip(labels.iter()).map(|(xy, &l)| {
                EmptyElement::at((xy[0], xy[1]))
                    + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(l)).filled())
                    + Text::new(format!("{}", l), (6, 0), ("sans-serif", 15.0).into_font())
            }))?;
        }
        None => {
            scatter_ctx.draw_series(data_values.chunks(2).map(|xy| {
                EmptyElement::at((xy[0], xy[1]))
                    + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(3)).filled())
            }))?;
        }
    }

    Ok(())
}

pub fn scatterplot_vec(data: Vec<f64>,
                       labels: &Vec<f64>,
                       title: &str) {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();

    let x_min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = data.iter().fold(f64::zero(), |a, &b| a.max(b));
    let y_min = x_min;
    let y_max = x_max + 1.0;

    root.fill(&WHITE).unwrap();
    let root = root.margin(15, 15, 15, 15);

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(x_min..x_max, y_min..y_max).unwrap();
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .draw().unwrap();
    let mut rng = rand::thread_rng();
    scatter_ctx.draw_series(data.chunks(1).zip(labels.iter()).map(|(x, &l)| {
        let jitter: f64 = rng.gen::<f64>().div(2.0);
        EmptyElement::at((x[0] + jitter, l + jitter))
            + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(l as usize)).filled())
            + Text::new(format!("{}", l), (6, 0), ("sans-serif", 15.0).into_font())
    })).expect("can't draw for some reason");
}


/// Get min value of `x` along axis `axis`
pub fn min<T: RealNumber>(x: &DenseMatrix<T>, axis: usize) -> T {
    let n = x.shape().0;
    x.slice(0..n, axis..axis + 1)
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

/// Get max value of `x` along axis `axis`
pub fn max<T: RealNumber>(x: &DenseMatrix<T>, axis: usize) -> T {
    let n = x.shape().0;
    x.slice(0..n, axis..axis + 1)
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

/// Generates 2x2 mesh grid from `x`
pub fn make_meshgrid(x: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let n = x.shape().0;
    let x_min = min(x, 0) - 1.0;
    let x_max = max(x, 0) + 1.0;
    let y_min = min(x, 1) - 1.0;
    let y_max = max(x, 1) + 1.0;

    let x_step = (x_max - x_min) / n as f64;
    let x_axis: Vec<f64> = (0..n).map(|v| (v as f64 * x_step) + x_min).collect();
    let y_step = (y_max - y_min) / n as f64;
    let y_axis: Vec<f64> = (0..n).map(|v| (v as f64 * y_step) + y_min).collect();

    let x_new: Vec<Vec<f64>> = x_axis
        .clone()
        .into_iter()
        .flat_map(move |v1| y_axis.clone().into_iter().map(move |v2| vec![v1, v2]))
        .collect();

    DenseMatrix::from_2d_vec(&x_new)
}

/// Draw a mesh grid defined by `mesh` with a scatterplot of `data` on top
/// We use Plotters library to draw scatter plot.
/// https://docs.rs/plotters/0.3.0/plotters/
pub fn scatterplot_with_mesh(
    mesh: &DenseMatrix<f64>,
    mesh_labels: &Vec<f64>,
    data: &DenseMatrix<f64>,
    labels: &Vec<f64>,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let root = root.margin(15, 15, 15, 15);

    let x_min = (min(mesh, 0) - 1.0) as f64;
    let x_max = (max(mesh, 0) + 1.0) as f64;
    let y_min = (min(mesh, 1) - 1.0) as f64;
    let y_max = (max(mesh, 1) + 1.0) as f64;

    let mesh_labels: Vec<usize> = mesh_labels.into_iter().map(|&v| v as usize).collect();
    let mesh: Vec<f64> = mesh.iter().map(|v| v as f64).collect();

    let labels: Vec<usize> = labels.into_iter().map(|&v| v as usize).collect();
    let data: Vec<f64> = data.iter().map(|v| v as f64).collect();

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;
    scatter_ctx.draw_series(mesh.chunks(2).zip(mesh_labels.iter()).map(|(xy, &l)| {
        EmptyElement::at((xy[0], xy[1]))
            + Circle::new((0, 0), 1, ShapeStyle::from(&Palette99::pick(l)).filled())
    }))?;
    scatter_ctx.draw_series(data.chunks(2).zip(labels.iter()).map(|(xy, &l)| {
        EmptyElement::at((xy[0], xy[1]))
            + Circle::new(
            (0, 0),
            3,
            ShapeStyle::from(&Palette99::pick(l + 3)).filled(),
        )
    }))?;

    Ok(())
}
