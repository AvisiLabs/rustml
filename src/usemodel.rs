use std::fs::File;
use std::io::{Read, Write};

use ndarray::{Array, Axis};
use smartcore::decomposition::pca::{PCA, PCAParameters};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::math::distance::euclidian;
use smartcore::metrics::{accuracy, mean_squared_error, roc_auc_score};
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_classifier::KNNClassifier;

use crate::smartcoreknn::{fill_dataset_return_y, get_dataset, get_dataset_kevin, get_tracks, make_meshgrid, scatterplot, scatterplot_vec, scatterplot_with_mesh};

mod smartcoreknn;

fn main() {
    let model = train_lr();
    {
        let bytes = bincode::serialize(&model).expect("Can not serialize the model");
        File::create("lr2.model")
            .and_then(|mut f| f.write_all(&bytes))
            .expect("Can not persist model");
    }
}

fn train_knn() {
    let (dataset, y) = get_dataset_kevin();
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
    println!("accuracy: {}", accuracy(&y_test, &predictions));

    scatterplot(
        &x_train,
        Some(&predictions.into_iter().map(|f| f as usize).collect()),
        "test",
    )
        .unwrap();
}

fn train_lr() -> LinearRegression<f64, DenseMatrix<f64>> {
    let (dataset, y) = get_dataset_kevin();
    let records = Array::from_shape_vec(dataset.raw_dim(), dataset.iter().cloned().collect()).unwrap();
    let x = DenseMatrix::from_array(
        y.len(),
        records.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let lr = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
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

    scatterplot_vec(
        predicvector,
        &y_test,
        "lr",
    );

    lr
}

fn print_accuracy() {
    let knn: KNNClassifier<f64, euclidian::Euclidian> = {
        let mut buf: Vec<u8> = Vec::new();
        File::open("knn.model")
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    };

    let (dataset, y) = get_dataset("marcel");
    let x = DenseMatrix::from_array(
        y.len(),
        dataset.len_of(Axis(1)),
        dataset.as_slice().unwrap(),
    );
    let prediction = knn.predict(&x).unwrap();
    println!("accuracy: {}", accuracy(&y, &prediction));
}
