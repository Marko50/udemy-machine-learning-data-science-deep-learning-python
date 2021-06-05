from tensorflow import feature_column
from tensorflow.keras.layers import DenseFeatures, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.data import Dataset
from tensorflow.keras.losses import BinaryCrossentropy
from dataset import X_train, y_train, X_test, y_test, dataframe


def df_to_ds(x, y, shuffle=True, batch_size=32):
    ds = Dataset.from_tensor_slices((dict(x), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size)
    return ds


def neural_network():
    age_col = feature_column.numeric_column('Age')
    shape_col = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Shape', dataframe['Shape'].unique()))
    margin_col = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Margin', dataframe['Margin'].unique()))
    density_col = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('Density', dataframe['Density'].unique()))
    feature_cols = [age_col, shape_col, margin_col, density_col]
    feature_layer = DenseFeatures(feature_cols)

    train_ds = df_to_ds(X_train, y_train)
    test_ds = df_to_ds(X_test, y_test)

    model = Sequential([
        feature_layer,
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(.1),
        Dense(1)
    ])

    model.compile(
            optimizer='adam',
            loss=BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10
    )
