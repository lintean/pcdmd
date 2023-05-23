from pydantic import BaseSettings

__all__ = ['AudioSettings', 'RawDatasetSettings', 'DatasetSettings', 'SETTINGS']


class CacheSettings(BaseSettings):
    cache_size: int = 128144


class AudioSettings(BaseSettings):
    sample_rate: int = 16000
    use_mono: bool = True


class TrainingSettings(BaseSettings):
    seed: int = 0
    #vocab: List[str] = ["yes","no","up","down","left","right","on","off","stop","go"]#['fire']
    num_epochs: int =50
    #num_labels: int = 35
    learning_rate: float = 2e-4#2e-4
    # device: str = str(util.get_gpu_with_max_memory(gpu_list))
    device: str = 'cuda:0'
    batch_size: int = 64
    lr_decay: float = 0.5
    max_window_size_seconds: float = 1
    eval_window_size_seconds: float = 1
    eval_stride_size_seconds: float = 0.063
    weight_decay: float = 0
    convert_static: bool = False
    objective: str = 'frame'  # frame or ctc
    token_type: str = 'word'
    phone_dictionary: str = None
    use_noise_dataset: bool = False


class RawDatasetSettings(BaseSettings):
    common_voice_dataset_path: str = None
    wake_word_dataset_path: str = None
    keyword_voice_dataset_path: str = None
    noise_dataset_path: str = None


class DatasetSettings(BaseSettings):
    dataset_path: str = None


class LazySettingsSingleton:
    _audio: AudioSettings = None
    _raw_dataset: RawDatasetSettings = None
    _dataset: DatasetSettings = None
    _cache: CacheSettings = None
    _training: TrainingSettings = None

    @property
    def audio(self) -> AudioSettings:
        if self._audio is None:
            self._audio = AudioSettings()
        return self._audio

    @property
    def raw_dataset(self) -> RawDatasetSettings:
        if self._raw_dataset is None:
            self._raw_dataset = RawDatasetSettings()
        return self._raw_dataset

    @property
    def dataset(self) -> DatasetSettings:
        if self._dataset is None:
            self._dataset = DatasetSettings()
        return self._dataset

    @property
    def cache(self) -> CacheSettings:
        if self._cache is None:
            self._cache = CacheSettings()
        return self._cache

    @property
    def training(self) -> TrainingSettings:
        if self._training is None:
            self._training = TrainingSettings()
        return self._training


SETTINGS = LazySettingsSingleton()
