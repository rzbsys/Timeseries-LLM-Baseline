import pandas as pd
from datetime import timedelta

SAMYANG_DEFAULT_TRAIN_COLUMNS = [
    # Stage1
    *[
        "Pol",
        "IS",
        "Moisture",
        "Ash",
        "Filterability",
        "Color",
        "Alcohol_Floc",
        "MELTER.MELTER_CONTROL_BX.Value",
        "MELTER_ML.TEMP_PV_Value",
    ],
    # Stage2
    *[
        "CO2.CO2_GAS_F.Value",
        "CO2.CO2_GAS_PRESS.Value",
        "CO2.CO2_GAS_concentration_Value",
        "SATURATOR1_PH_PV.Value",
        "SATURATOR1_ST_TEMP.Value",
        "SATURATOR1_CALCIUM_F.Value",
        "SATURATOR1_RATE_SV.Value",
        "SATURATOR2_PH_PV.Value",
        "SATURATOR2_ST_TEMP.Value",
        "SATURATOR2_CALCIUM_F.Value",
        "Saturator1_exp_CaO",
        "Saturator2_exp_CaO",
    ],
    # Stage3
    *[
        "SATURATOR3_PH_PV.Value",
        "SATURATOR3_ST_TEMP.Value",
        "Saturator3_exp_CaO",
        "SATURATOR_ML_SUPPLY_F_PV.Value",
    ],
]

SAMYANG_DEFAULT_TARGET_COLUMNS = "SATURATOR_ML_SUPPLY_F_PV.Value"


# 아래는 시연님이 보내주신 Dataloader에서 전처리 하는 함수를 따로 추출 한 영역임.
def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["outlier"] != 1]


def filter_by_date(data: pd.DataFrame, date: str = "2021-04-30") -> pd.DataFrame:
    return data[data["TimeStamp"] > date].reset_index(drop=True)


SAMYANG_DEFAULT_PREPROCESS_FNS = [
    remove_outliers,
    filter_by_date,
]


def validate_timestamp(patch: pd.DataFrame) -> bool:
    minute_interval = 15
    timestamp = (patch["TimeStamp"].iloc[-1] - patch["TimeStamp"].iloc[0]).seconds

    expected_interval = [
        timedelta(minutes=minute_interval * (len(patch) - 1)).seconds,
        timedelta(minutes=minute_interval).seconds,
    ]

    return timestamp in expected_interval


SAMYANG_DEFAULT_PATCH_VALIDATE_FNS = [
    validate_timestamp,
]



from data.manufacturing.generate_patch import generate_patch_report as generate_samyang_report
SAMYANG_DEFAULT_REPORT_FN = generate_samyang_report



# EEG 데이터
BIOSIGNAL_DEFAULT_TRAIN_COLUMNS = [
    "C3",
    "C4",
    "CZ",
    "F3",
    "F4",
    "F7",
    "F8",
    "FP1",
    "FP2",
    "FZ",
    "O1",
    "O2",
    "P3",
    "P4",
    "PZ",
    "T3",
    "T4",
    "T5",
    "T6",
]
BIOSIGNAL_DEFAULT_TARGET_COLUMNS = "label"

BIOSIGNAL_DEFAULT_PREPROCESS_FNS = []


def validate_same_sub(patch: pd.DataFrame) -> bool:
    patch_subs = patch["sub_id"].unique()
    return len(patch_subs) == 1


BIOSIGNAL_DEFAULT_PATCH_VALIDATE_FNS = [
    validate_same_sub,
]


from data.biosignal.generate_patch import generate_patch_report as generate_biosignal_report
BIOSIGNAL_DEFAULT_REPORT_FN = generate_biosignal_report
