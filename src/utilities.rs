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
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::math::num::RealNumber;
// Model performance
use smartcore::metrics::mean_squared_error;
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
        let vector = Vec::from([
            self.danceability,
            self.energy,
            self.key.to_f64().unwrap(),
            self.loudness,
            self.mode.to_f64().unwrap(),
            self.speechiness,
            self.instrumentalness,
            self.liveness,
            self.valence,
            self.tempo,
            self.time_signature.to_f64().unwrap(),
        ]);
        return Array::from(arr1(&*vector));
    }
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
    for rowerik in get_tracks("erik")
        .expect("can't get dataset for erik")
        .rows()
    {
        if i < 54 {
            i += 1;
            data_array.push_row(rowerik).unwrap();
        }
    }
    targets.append(&mut vec![1.0; data_array.nrows() - targets.len()]);
    return (data_array, targets);
}

pub fn fill_dataset_return_y(array: &mut Array2<f64>) -> Vec<f64> {
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
    let mut conn = Client::connect(
        "postgresql://admin:fkdvaQmAK82a4cGfdZSQ8rH6cyE2gCac@localhost/rustspotify",
        NoTls,
    )
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

pub fn to_array2<T: Copy>(source: &[Array1<T>]) -> Result<Array2<T>, impl std::error::Error> {
    let width = source.len();
    let flattened: Array1<T> = source.into_iter().flat_map(|row| row.to_vec()).collect();
    let height = flattened.len() / width;
    flattened.into_shape((width, height))
}

/// Draw a scatterplot of `data` with labels `labels`
/// We use Plotters library to draw scatter plot.
/// https://docs.rs/plotters/0.3.0/plotters/
#[allow(dead_code)]
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
    scatter_ctx.configure_mesh().disable_x_mesh().draw()?;
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

pub fn performance_graph(
    path: &str,
    caption: &str,
    data: Vec<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(path, (640, 480)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption(caption, ("sans-serif", 50.0))
        .build_cartesian_2d(0..2, 0..100)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(BLUE.mix(0.7).filled())
            .data(data.iter().enumerate().map(|(i, x)| (i as i32, *x))),
    )?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", path);

    Ok(())
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
