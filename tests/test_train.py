import pandas as pd


def test_dataset_exists():
    df = pd.read_csv("data/diabetes.csv")

    assert not df.empty


def test_dataset_columns():
    df = pd.read_csv("data/diabetes.csv")

    assert "glucose" in df.columns
    assert "bloodpressure" in df.columns
    assert "diabetes" in df.columns


def test_target_values():
    df = pd.read_csv("data/diabetes.csv")

    assert set(df["diabetes"].unique()).issubset({0, 1})
