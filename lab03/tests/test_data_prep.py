import pandas as pd
import pytest

from data_prep import load_data, split_data


@pytest.fixture
def df():
    # maly zbior testowy: 100 probek, 2 cechy, 2 zbalansowane klasy
    return pd.DataFrame({
        "x1": range(100),
        "x2": [i * 0.5 for i in range(100)],
        "target": ["a", "b"] * 50,
    })


# Zadanie 5 - testy jednostkowe funkcji z data_prep.py
@pytest.mark.parametrize("test_size", [0.2, 0.25, 0.3])
def test_split_sizes(df, test_size):
    X_train, X_test, y_train, y_test = split_data(df, test_size, seed=42)
    assert len(X_test) == pytest.approx(len(df) * test_size, abs=1)
    assert len(y_test) == len(X_test)


def test_split_sum_equals_whole(df):
    X_train, X_test, y_train, y_test = split_data(df, 0.2, seed=42)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_split_removes_target_from_features(df):
    X_train, X_test, _, _ = split_data(df, 0.2, seed=42)
    assert list(X_train.columns) == ["x1", "x2"]
    assert list(X_test.columns) == ["x1", "x2"]


def test_split_keeps_class_proportions(df):
    _, _, y_train, y_test = split_data(df, 0.2, seed=42)
    assert y_train.value_counts().to_dict() == {"a": 40, "b": 40}
    assert y_test.value_counts().to_dict() == {"a": 10, "b": 10}


def test_split_is_reproducible(df):
    # evaluate.py odtwarza zbior testowy z train.py tylko dzieki temu samemu ziarnu
    first = split_data(df, 0.2, seed=42)
    second = split_data(df, 0.2, seed=42)
    for a, b in zip(first, second):
        assert a.index.equals(b.index)


def test_split_empty_dataframe():
    with pytest.raises(ValueError, match="Pusty"):
        split_data(pd.DataFrame(columns=["x1", "target"]), 0.2, seed=42)


def test_split_missing_target(df):
    with pytest.raises(ValueError, match="target"):
        split_data(df.drop(columns="target"), 0.2, seed=42)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5])
def test_split_invalid_test_size(df, test_size):
    with pytest.raises(ValueError, match="test_size"):
        split_data(df, test_size, seed=42)


def test_load_data_reads_csv(tmp_path, df):
    path = tmp_path / "dane.csv"
    df.to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_data(path), df)


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "brak.csv")
