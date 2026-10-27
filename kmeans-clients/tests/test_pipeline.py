from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.pipeline import (  # noqa: E402
    DATA_PATH,
    RAW_COLUMNS,
    RFM_FEATURES,
    RFMFeatureBuilder,
    CustomerPreprocessor,
    make_pipeline,
)


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH).head(80).copy()
    return frame[RAW_COLUMNS]


def test_rfm_builder_creates_expected_features(raw_sample: pd.DataFrame) -> None:
    builder = RFMFeatureBuilder().fit(raw_sample)
    rfm = builder.transform(raw_sample)
    assert list(rfm.columns) == RFM_FEATURES
    expected_frequency = raw_sample[
        ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    ].sum(axis=1)
    expected_monetary = raw_sample[
        ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    ].sum(axis=1)
    assert np.allclose(rfm["Frequency"], expected_frequency)
    assert np.allclose(rfm["Monetary"], expected_monetary)
    assert (rfm["Tenure"] == 2014 - pd.to_datetime(raw_sample.Dt_Customer, dayfirst=True).dt.year).all()


def test_pipeline_fit_predict_and_no_nan(raw_sample: pd.DataFrame) -> None:
    pipeline = make_pipeline(n_clusters=3, seed=42).fit(raw_sample)
    predictions = pipeline.predict(raw_sample)
    transformed = pipeline[:-1].transform(raw_sample)
    assert len(predictions) == len(raw_sample)
    assert set(np.unique(predictions)) <= {0, 1, 2}
    assert np.isfinite(transformed).all()
    assert not pd.isna(predictions).any()


def test_campaign_outcomes_do_not_change_predictions(raw_sample: pd.DataFrame) -> None:
    pipeline = make_pipeline(n_clusters=3, seed=42).fit(raw_sample)
    with_outcomes = raw_sample.copy()
    with_outcomes["Response"] = np.arange(len(raw_sample)) % 2
    with_outcomes["AcceptedCmp1"] = 1 - with_outcomes["Response"]
    assert np.array_equal(pipeline.predict(raw_sample), pipeline.predict(with_outcomes))


def test_preprocessor_imputes_income_without_nan(raw_sample: pd.DataFrame) -> None:
    engineered = RFMFeatureBuilder().fit_transform(raw_sample)
    engineered.loc[engineered.index[:3], "Income"] = np.nan
    processor = CustomerPreprocessor().fit(engineered)
    transformed = processor.transform(engineered)
    assert np.isfinite(transformed).all()


def test_invalid_negative_recency_is_rejected(raw_sample: pd.DataFrame) -> None:
    invalid = raw_sample.copy()
    invalid.loc[invalid.index[0], "Recency"] = -1
    with pytest.raises(ValueError, match="positives ou nulles"):
        RFMFeatureBuilder().fit(invalid)
