use std::fs::File;
use std::io::{Read, Write};

use ndarray::{arr1, Array, Array1, Axis};
use num_traits::ToPrimitive;
use serde::Serialize;
use smartcore::cluster::dbscan::{DBSCAN, DBSCANParameters};
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::math::distance::euclidian;
use smartcore::math::distance::euclidian::Euclidian;
use smartcore::metrics::{accuracy, completeness_score, f1, homogeneity_score, mean_squared_error, recall, roc_auc_score, v_measure_score};
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};

use crate::utilities::{get_dataset, get_dataset_kevin, performance_graph, scatterplot, to_array2};

mod utilities;

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

fn main() {
    let dt: DecisionTreeClassifier<f64> = {
        let mut buf: Vec<u8> = Vec::new();
        File::open("dt.model")
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    };

    let track = Track {
        danceability: 0.382,
        energy: 0.354,
        key: 7,
        loudness: -6.595,
        mode: 1,
        speechiness: 0.0289,
        instrumentalness: 3.11e-06,
        liveness: 0.168,
        valence: 0.435,
        tempo: 87.018,
        time_signature: 4
    };
    let array = track.to_array();
    let vec = vec!(array);
    let array2d = to_array2(vec.as_slice()).unwrap();
    let x = DenseMatrix::from_array(
        1,
        array2d.len_of(Axis(1)),
        array2d.as_slice().unwrap(),
    );
    let prediction = dt.predict(&x).unwrap();
    println!("{:?}", prediction);
}

#[allow(dead_code)]
fn train_knn() {
    let (dataset, y) = get_dataset_kevin();
    let records =
        Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let knn = KNNClassifier::fit(&x_train, &y_train, Default::default()).unwrap();
    let predictions = knn.predict(&x_test).unwrap();
    let predict_train = knn.predict(&x_train).unwrap();
    println!("MSE: {}", mean_squared_error(&y_test, &predictions));
    println!("accuracy: {}", accuracy(&y_test, &predictions));

    // scatterplot(
    //     &x_train,
    //     Some(&predictions.into_iter().map(|f| f as usize).collect()),
    //     "test",
    // )
    //     .unwrap();

    let test_acc = accuracy(&y_test, &predictions);

    let train_acc = accuracy(&y_train, &predict_train);
    let accuracy = vec![(test_acc * 100.0f64) as i32, (train_acc * 100.0f64) as i32];
    println!("{:?}", accuracy);
    performance_graph("knn_accuracy.svg", "accuracy", accuracy);

    let test_f1 = f1(&y_test, &predictions, 1.0);

    let train_f1 = f1(&y_train, &predict_train, 1.0);
    let f1 = vec![(test_f1 * 100.0f64) as i32, (train_f1 * 100.0f64) as i32];
    println!("{:?}", f1);
    performance_graph("knn_f1.svg", "F1 Score", f1);

    let data = recall(&y_test, &predictions);

    let compare = recall(&y_train, &predict_train);
    let f1 = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", f1);
    performance_graph("kn_recall.svg", "Recall Score", f1)
        .expect("Write of recall graph was not successful");
}

fn train_dt() -> DecisionTreeClassifier<f64> {
    let (dataset, y) = get_dataset_kevin();
    let records =
        Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let dt = DecisionTreeClassifier::fit(
        &x_train,
        &y_train,
        DecisionTreeClassifierParameters::default(),
    )
        .unwrap();
    let predictions = dt.predict(&x_test).unwrap();
    let predict_train = dt.predict(&x_train).unwrap();

    let test_acc = accuracy(&y_test, &predictions);

    let train_acc = accuracy(&y_train, &predict_train);
    let accuracy = vec![(test_acc * 100.0f64) as i32, (train_acc * 100.0f64) as i32];
    println!("{:?}", accuracy);
    performance_graph("dt_accuracy.svg", "accuracy", accuracy)
        .expect("Write of accuracy graph was not successful");

    let test_f1 = f1(&y_test, &predictions, 1.0);

    let train_f1 = f1(&y_train, &predict_train, 1.0);
    let f1 = vec![(test_f1 * 100.0f64) as i32, (train_f1 * 100.0f64) as i32];
    println!("{:?}", f1);
    performance_graph("dt_f1.svg", "F1 Score", f1)
        .expect("Write of f1 graph was not successful");

    let data = recall(&y_test, &predictions);

    let compare = recall(&y_train, &predict_train);
    let f1 = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", f1);
    performance_graph("dt_recall.svg", "Recall Score", f1)
        .expect("Write of recall graph was not successful");
    dt
}

#[allow(dead_code)]
fn train_rf() -> RandomForestClassifier<f64> {
    let (dataset, y) = get_dataset_kevin();
    let records =
        Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let rf = RandomForestClassifier::fit(
        &x_train,
        &y_train,
        RandomForestClassifierParameters::default(),
    )
        .unwrap();
    let predictions = rf.predict(&x_test).unwrap();
    let predict_train = rf.predict(&x_train).unwrap();
    println!("MSE: {}", mean_squared_error(&y_test, &predictions));
    println!("accuracy: {}", accuracy(&y_test, &predictions));

    // scatterplot(
    //     &x_train,
    //     Some(&predictions.into_iter().map(|f| f as usize).collect()),
    //     "test",
    // )
    //     .unwrap();

    let test_acc = accuracy(&y_test, &predictions);

    let train_acc = accuracy(&y_train, &predict_train);
    let accuracy = vec![(test_acc * 100.0f64) as i32, (train_acc * 100.0f64) as i32];
    println!("{:?}", accuracy);
    performance_graph("rf_accuracy.svg", "accuracy", accuracy)
        .expect("Write of accuracy graph was not successful");

    let data = f1(&y_test, &predictions, 1.0);

    let compare = f1(&y_train, &predict_train, 1.0);
    let f1 = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", f1);
    performance_graph("rf_f1.svg", "F1 Score", f1)
        .expect("Write of f1 graph was not successful");

    let data = recall(&y_test, &predictions);

    let compare = recall(&y_train, &predict_train);
    let f1 = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", f1);
    performance_graph("rf_recall.svg", "Recall Score", f1)
        .expect("Write of recall graph was not successful");
    rf
}

#[allow(dead_code)]
fn train_lir() -> LinearRegression<f64, DenseMatrix<f64>> {
    let (dataset, y) = get_dataset_kevin();
    let records =
        Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let lr = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    let predictions = lr.predict(&x_test).unwrap();
    let predictions_training = lr.predict(&x_train).unwrap();

    let mut predict_test: Vec<f64> = Vec::new();
    let mut i = 0;
    for float in predictions.as_slice() {
        predict_test.insert(i as usize, f64::round(*float));
        i += 1;
    }

    let mut predict_train: Vec<f64> = Vec::new();
    let mut i = 0;
    for float in predictions_training.as_slice() {
        predict_train.insert(i as usize, f64::round(*float));
        i += 1;
    }

    println!("{:?}", predict_test);
    println!("roc auc: {}", roc_auc_score(&y_test, &predictions));
    println!("accuracy: {}", accuracy(&y_test, &predict_test));
    let data = accuracy(&y_test, &predict_test);

    let compare = accuracy(&y_train, &predict_train);
    let accuracy = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", accuracy);
    performance_graph("lin_accuracy.svg", "accuracy", accuracy)
        .expect("Write of accuracy graph was not successful");

    let data = f1(&y_test, &predict_test, 1.0);

    let compare = f1(&y_train, &predict_train, 1.0);
    let f1 = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", f1);
    performance_graph("lin_f1.svg", "F1 Score", f1)
        .expect("Write of f1 graph was not successful");

    let data = recall(&y_test, &predict_test);

    let compare = recall(&y_train, &predict_train);
    let recall = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", recall);
    performance_graph("lin_recall.svg", "Recall Score", recall)
        .expect("Write of recall graph was not successful");

    lr
}

#[allow(dead_code)]
fn train_lor() -> LogisticRegression<f64, DenseMatrix<f64>> {
    let (dataset, y) = get_dataset_kevin();
    let records =
        Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let lr = LogisticRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    let predictions = lr.predict(&x_test).unwrap();

    let mut predicvector: Vec<f64> = Vec::new();
    let mut i = 0;
    for float in predictions.as_slice() {
        predicvector.insert(i as usize, f64::round(*float));
        i += 1;
    }

    println!("{:?}", predicvector);
    println!("roc auc: {}", roc_auc_score(&y_test, &predictions));
    println!("accuracy: {}", accuracy(&y_test, &predicvector));

    // scatterplot_vec(
    //     predicvector,
    //     &y_test,
    //     "lor",
    // );

    let data = accuracy(&y_test, &predictions);

    let compare = accuracy(&y_train, &lr.predict(&x_train).unwrap());
    let accuracy = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", accuracy);
    performance_graph("log_accuracy.svg", "accuracy", accuracy)
        .expect("Write of accuracy graph was not successful");

    let data = f1(&y_test, &predictions, 1.0);

    let compare = f1(&y_train, &lr.predict(&x_train).unwrap(), 1.0);
    let f1 = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    performance_graph("log_f1.svg", "F1 Score", f1)
        .expect("Write of f1 graph was not successful");

    let data = recall(&y_test, &predictions);
    let compare = recall(&y_train, &lr.predict(&x_train).unwrap());
    let recall = vec![(data * 100.0f64) as i32, (compare * 100.0f64) as i32];
    println!("{:?}", recall);
    performance_graph("log_recall.svg", "Recall Score", recall)
        .expect("Write of recall graph was not successful");
    lr
}

#[allow(dead_code)]
fn kmeans() -> KMeans<f64> {
    let (dataset, y) = get_dataset_kevin();
    let records =
        Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (_x_train, _x_test, _y_train, _y_test) = train_test_split(&x, &y, 0.2, true);
    // These are our target class labels
    // Fit & predict
    let kmeans = KMeans::fit(&x, KMeansParameters::default().with_k(7)).unwrap();
    let predictions = kmeans.predict(&x).unwrap();

    // Measure performance
    println!("Homogeneity: {}", homogeneity_score(&y, &predictions));
    println!("Completeness: {}", completeness_score(&y, &predictions));
    println!("V Measure: {}", v_measure_score(&y, &predictions));

    scatterplot(
        &x,
        Some(&predictions.into_iter().map(|f| f as usize).collect()),
        "kmeansclusters",
    ).expect("couldn't write graph kmeans");

    kmeans
}

#[allow(dead_code)]
fn dbscan() -> DBSCAN<f64, Euclidian> {
    let (dataset, y) = get_dataset_kevin();
    let records =
        Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (_x_train, _x_test, _y_train, _y_test) = train_test_split(&x, &y, 0.2, true);
    // These are our target class labels
    // Fit & predict
    let dbscan = DBSCAN::fit(
        &x,
        DBSCANParameters::default()
            .with_eps(0.2)
            .with_min_samples(7),
    )
        .unwrap();
    let predictions = dbscan.predict(&x).unwrap();

    // Measure performance
    println!("Homogeneity: {}", homogeneity_score(&y, &predictions));
    println!("Completeness: {}", completeness_score(&y, &predictions));
    println!("V Measure: {}", v_measure_score(&y, &predictions));

    scatterplot(
        &x,
        Some(&predictions.into_iter().map(|f| f as usize).collect()),
        "dbscan",
    ).expect("couldn't write graph dbscan");

    dbscan
}

fn write_model<T: Serialize>(model: T, path: &str) {
    let bytes = bincode::serialize(&model).expect("Can not serialize the model");
    File::create(path)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("Can not persist model");
}
