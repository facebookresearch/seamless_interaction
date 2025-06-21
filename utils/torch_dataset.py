from abc import abstractmethod

from torch.utils.data import Dataset


def load_jsonl(
    jsonl_path: Path, feature_dir: Path
) -> List[Tuple[Dict[str, Any], Path]]:
    """Load a single jsonl file
    Args:
        jsonl_path: Path to the jsonl file
    """
    with open(jsonl_path, "r") as f:
        data = [(json.loads(line), feature_dir) for line in f]
    return data


# TODO: remove duplicate code for methods that appear in both the torch_dataset and fs2_dataset
class TorchSeamlessCommunicationDataset(Dataset):
    def __init__(
        self,
        jsonl_list: List[str],
        jsonl_weights: List[float],
        feature_list: List[str],
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
        feature_dir_list: Optional[List[str]] = None,
        mismatch_tolerance_seconds: float = DEFAULT_MISMATCH_TOLERANCE_SECONDS,
    ):
        if len(jsonl_list) != len(jsonl_weights):
            raise ConfigurationError(
                f"Found differing jsonl_list and jsonl_weights lengths: {len(jsonl_list)} != {len(jsonl_weights)}"
            )
        self.jsonl_list = jsonl_list
        self.jsonl_weights = jsonl_weights
        self.feature_dict = self._prepare_feature_dict(feature_list)
        self.transform_fns = self._prepare_transforms(transforms, transforms_kwargs)
        if feature_dir_list is not None:
            if len(feature_dir_list) != len(jsonl_list):
                raise ConfigurationError(
                    f"Found differing feature_dir_list and jsonl_list lengths: {len(feature_dir_list)} != {len(jsonl_list)}"
                )
            self.feature_dir_list = feature_dir_list
        else:
            self.feature_dir_list = [
                str(Path(jsonl_path).parent) for jsonl_path in jsonl_list
            ]
        self.mismatch_tolerance_seconds = mismatch_tolerance_seconds

        self.data = []
        for jsonl_path, feature_dir in zip(self.jsonl_list, self.feature_dir_list):
            if not Path(jsonl_path).exists():
                raise FileNotFoundError(f"JSONL file {jsonl_path} does not exist!")
            if not Path(feature_dir).exists():
                raise FileNotFoundError(
                    f"Feature directory {feature_dir} does not exist!"
                )
            self.data += self.load_jsonl(jsonl_path, Path(feature_dir))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Load the features for sample at index
        sample, feature_basepath = self.data[index]
        sample_features = self._load_sample(sample, feature_basepath)
        # Transform the  data
        for transform_fn in self.transform_fns:
            sample_features = transform_fn(sample_features)
            if isinstance(sample_features, list):
                # TODO: same as jsonl_builder.yield_from read_sequence.and_return
                pass
        return sample_features

    def _prepare_feature_dict(self, feature_list: List[str]) -> Dict[str, Set[str]]:
        """Prepare the feature dict for the dataset. Makes sure that each requested feature follows the valid
        formatting convention <collection>:<feature_name> and prepares a dict to load feature_name keys from each h5py collection.
        Args:
            feature_list: List of features to load
        """
        feature_dict: Dict[str, Set[str]] = {}
        for feature in feature_list:
            split_feature_name = feature.split(":")
            if len(split_feature_name) != 2:
                raise ConfigurationError(
                    f"Provided feature name {feature} which does not fit <collection>:<feature_name> convention!"
                )
            collection, feature_name = split_feature_name
            validate_feature_name(collection, feature_name)
            feature_dict.setdefault(collection, set()).add(feature_name)
        return feature_dict

    def _prepare_transforms(
        self,
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
    ) -> List[partial[Any]]:
        """Prepare the transforms for the dataset. Make sure that the transforms are callable and the kwargs are dictionaries
        Args:
            transforms: List of transforms to apply to each sample
            transforms_kwargs: List of kwargs to pass to the transform fns
        """
        if len(transforms) != len(transforms_kwargs):
            raise ConfigurationError(
                f"Found differing lengths for transforms and transforms_kwargs ({len(transforms)} != {len(transforms_kwargs)})"
            )

        transform_fns = []
        for transform, kwargs in zip(transforms, transforms_kwargs):
            if not callable(transform):
                raise ConfigurationError(f"Transform {transform} is not callable")
            if not isinstance(kwargs, dict):
                raise ConfigurationError(
                    f"Transform kwargs {kwargs} is not a dictionary"
                )
            try:
                transform_fns.append(partial(transform, **kwargs))
            except Exception as e:
                raise DatasetError(
                    f"Failed to create partial function for transform {transform}: {e}"
                )
        return transform_fns

    def _load_feature_set(
        self,
        sample: Dict[str, Any],
        collection: str,
        features: Set[str],
        feature_basepath: Path,
    ) -> Dict[str, Any]:
        """Load a single feature set for a sample from a jsonl file
        Args:
            sample: Sample to load
            collection: name of feature collection to load (ie smirk, metadata, smplh etc.)
            features: list of feature key strings to load from each h5py collection
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """
        load_feature_fn = get_load_feature_fn(feature_basepath, collection, sample)
        feature_data: Dict[str, Any] = load_feature_fn(features=features)
        return feature_data

    @abstractmethod
    def _load_sample(
        self, sample: Dict[str, Any], feature_basepath: Path
    ) -> Dict[str, Any]:
        """Load a single sample from the jsonl file (one line)
        Args:
            sample: Sample to load
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """


class TorchSeamlessCommunicationMondadicDataset(Dataset):
    def __init__(
        self,
        jsonl_list: List[str],
        jsonl_weights: List[float],
        feature_list: List[str],
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
        feature_dir_list: Optional[List[str]] = None,
        mismatch_tolerance_seconds: float = DEFAULT_MISMATCH_TOLERANCE_SECONDS,
    ):
        super().__init__(
            jsonl_list,
            jsonl_weights,
            feature_list,
            transforms,
            transforms_kwargs,
            feature_dir_list,
        )

    @override
    def _load_sample(
        self, sample: Dict[str, Any], feature_basepath: Path
    ) -> Dict[str, Any]:
        """Load a single monadic sample from the jsonl file (one line)
        Args:
            sample: Sample to load
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """
        sample_id = sample[JSONL_KEY_ID]
        sample_features = {}
        for collection, features in self.feature_dict.items():
            sample_features.update(
                self._load_feature_set(sample, collection, features, feature_basepath)
            )

        # Assert that the durations for all features are roughly equal
        sample_features = validate_feature_duration_consistency(
            sample_id, sample_features, self.mismatch_tolerance_seconds
        )
        return {
            **sample_features,
            JSONL_KEY_ID: sample_id,
            JSONL_KEY_VISUAL_RATE: sample[JSONL_KEY_VISUAL_RATE],
            JSONL_KEY_SPEECH_TOKEN_RATE: sample[JSONL_KEY_SPEECH_TOKEN_RATE],
        }


class TorchSeamlessCommunicationDyadicDataset(Dataset):
    def __init__(
        self,
        jsonl_list: List[str],
        jsonl_weights: List[float],
        feature_list: List[str],
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
        feature_dir_list: Optional[List[str]] = None,
        mismatch_tolerance_seconds: float = DEFAULT_MISMATCH_TOLERANCE_SECONDS,
    ):
        super().__init__(
            jsonl_list,
            jsonl_weights,
            feature_list,
            transforms,
            transforms_kwargs,
            feature_dir_list,
        )

    @override
    def _load_sample(
        self, sample: Dict[str, Any], feature_basepath: Path
    ) -> Dict[str, Any]:
        """Load a single dyadic sample from the jsonl file (one line)
        Args:
            sample: Sample to load
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """
        sample_spkr0 = sample[DYADIC_JSONL_KEY_ID_0]
        sample_spkr1 = sample[DYADIC_JSONL_KEY_ID_1]
        sample_features_spkr0 = {}
        sample_features_spkr1 = {}
        for collection, features in self.feature_dict.items():
            sample_features_spkr0.update(
                self._load_feature_set(
                    sample_spkr0, collection, features, feature_basepath
                )
            )
            sample_features_spkr1.update(
                self._load_feature_set(
                    sample_spkr1, collection, features, feature_basepath
                )
            )

        # Assert that the durations for all features are roughly equal
        sample_features_spkr0 = validate_feature_duration_consistency(
            sample_spkr0[JSONL_KEY_ID],
            sample_features_spkr0,
            self.mismatch_tolerance_seconds,
        )
        sample_features_spkr1 = validate_feature_duration_consistency(
            sample_spkr1[JSONL_KEY_ID],
            sample_features_spkr1,
            self.mismatch_tolerance_seconds,
        )
        return {
            DYADIC_JSONL_KEY_ID_0: {
                **sample_features_spkr0,
                JSONL_KEY_ID: sample_spkr0[JSONL_KEY_ID],
                JSONL_KEY_VISUAL_RATE: sample_spkr0[JSONL_KEY_VISUAL_RATE],
                JSONL_KEY_SPEECH_TOKEN_RATE: sample_spkr0[JSONL_KEY_SPEECH_TOKEN_RATE],
            },
            DYADIC_JSONL_KEY_ID_1: {
                **sample_features_spkr1,
                JSONL_KEY_ID: sample_spkr1[JSONL_KEY_ID],
                JSONL_KEY_VISUAL_RATE: sample_spkr1[JSONL_KEY_VISUAL_RATE],
                JSONL_KEY_SPEECH_TOKEN_RATE: sample_spkr1[JSONL_KEY_SPEECH_TOKEN_RATE],
            },
        }
