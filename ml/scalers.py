from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler


def create_scaler(normalizer: str):
    normalizer = normalizer.lower()

    if normalizer in ["standard", "standardscaler"]:
        return StandardScaler()

    if normalizer in ["robust", "robustscaler"]:
        return RobustScaler()

    if normalizer in ["minmax", "minmaxscaler"]:
        return MinMaxScaler()

    if normalizer in ["maxabs", "maxabsscaler"]:
        return MaxAbsScaler()

    if normalizer in ["none", "no", "false"]:
        return None

    raise ValueError(
        "normalizer must be one of: standard, robust, minmax, maxabs, none"
    )