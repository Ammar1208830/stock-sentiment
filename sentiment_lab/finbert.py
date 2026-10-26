"""Pinned, local FinBERT inference. No replacement scorers or remote code."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re

from .contracts import ContractError, nonempty
from .io import file_hash, write_json
from .store import digest


MODEL_ID = 'ProsusAI/finbert'
REVISION = '0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031'
MODEL_FILES = ('config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt', 'README.md')
LABELS = ('positive', 'negative', 'neutral')
PREPROCESSING_VERSION = 'whitespace-v1'


@dataclass(frozen=True)
class SentimentConfig:
    schema_version: int = 1
    model_id: str = MODEL_ID
    revision: str = REVISION
    batch_size: int = 8
    max_length: int = 512
    device: str = 'cpu'
    cpu_threads: int = 2
    seed: int = 42

    @classmethod
    def load(cls, path: Path):
        import tomllib
        with path.open('rb') as stream:
            raw = tomllib.load(stream)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ContractError(f'Unknown sentiment settings: {sorted(unknown)}')
        result = cls(**raw)
        result.validate()
        return result

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ContractError('Sentiment schema_version must be 1')
        if self.model_id != MODEL_ID or not isinstance(self.revision, str) or not re.fullmatch(r'[a-f0-9]{40}', self.revision):
            raise ContractError('Use ProsusAI/finbert and a full 40-character commit revision')
        for key, low, high in (('batch_size', 1, 64), ('max_length', 8, 512), ('cpu_threads', 1, 16), ('seed', 0, 2**32 - 1)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ContractError(f'{key} must be an integer in [{low}, {high}]')
        if self.device != 'cpu':
            raise ContractError('M3 uses CPU inference for the reproducible reference implementation')


def normalize_text(text: str) -> str:
    text = nonempty(text, 'sentiment text')
    if len(text) > 100000:
        raise ContractError('Sentiment input exceeds the 100,000-character safety limit')
    return ' '.join(text.split())


def label_mapping(raw: dict) -> dict[int, str]:
    if not isinstance(raw, dict):
        raise ContractError('FinBERT id2label must be an object')
    try:
        result = {int(index): value.lower() for index, value in raw.items()}
    except (ValueError, TypeError, AttributeError) as exc:
        raise ContractError('Invalid FinBERT label mapping') from exc
    if set(result) != {0, 1, 2} or set(result.values()) != set(LABELS) or len(raw) != 3:
        raise ContractError('FinBERT must map exactly three indices to positive, negative and neutral')
    return result


def validate_prediction(row: dict) -> dict:
    if not isinstance(row, dict):
        raise ContractError('Sentiment prediction must be an object')
    values = {}
    for label in LABELS:
        value = row.get(label)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ContractError('Sentiment probabilities must be finite numbers in [0, 1]')
        values[label] = float(value)
    if not math.isclose(sum(values.values()), 1, abs_tol=1e-5):
        raise ContractError('Sentiment probabilities must sum to one')
    if type(row.get('token_count')) is not int or row['token_count'] <= 0 or type(row.get('truncated')) is not bool:
        raise ContractError('Prediction requires a positive original token count and boolean truncation flag')
    return {**values, 'polarity': values['positive'] - values['negative'],
            'label': max(LABELS, key=values.get), 'token_count': row['token_count'], 'truncated': row['truncated']}


def prepare_model(settings: SentimentConfig, model_root: Path, offline: bool = False) -> dict:
    settings.validate()
    directory = model_root / settings.revision
    receipt = directory / 'model_manifest.json'
    if not receipt.exists():
        if offline:
            raise ContractError('Pinned model is not available locally. Run prepare-finbert once with network access.')
        try:
            os.environ.setdefault('HF_XET_CACHE', str((model_root / '.xet').resolve()))
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ContractError('Install requirements-sentiment.lock to download FinBERT') from exc
        directory.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=settings.model_id, revision=settings.revision, local_dir=directory,
                              allow_patterns=list(MODEL_FILES), token=False, max_workers=2)
        except Exception as exc:
            raise ContractError(f'Pinned FinBERT download failed ({type(exc).__name__}); retry to resume') from None
        if any(not (directory / name).is_file() for name in MODEL_FILES):
            raise ContractError('Pinned model download is incomplete; all model/tokenizer files are required')
        config = json.loads((directory / 'config.json').read_text())
        mapping = label_mapping(config.get('id2label'))
        metadata = {'model_id': settings.model_id, 'revision': settings.revision,
                    'files_sha256': {name: file_hash(directory / name) for name in MODEL_FILES},
                    'id2label': mapping, 'weights_format': 'safetensors',
                    'source': f'https://huggingface.co/{settings.model_id}/tree/{settings.revision}',
                    'license_reference': f'https://huggingface.co/{settings.model_id}/blob/{settings.revision}/README.md'}
        write_json(receipt, metadata)
    try:
        metadata = json.loads(receipt.read_text())
        if metadata['model_id'] != settings.model_id or metadata['revision'] != settings.revision or set(metadata['files_sha256']) != set(MODEL_FILES):
            raise ContractError('Model receipt differs from the requested pinned artifacts')
        for name, expected in metadata['files_sha256'].items():
            if not (directory / name).is_file() or file_hash(directory / name) != expected:
                raise ContractError(f'Local model artifact is missing or changed: {name}')
        actual_mapping = label_mapping(json.loads((directory / 'config.json').read_text()).get('id2label'))
        if actual_mapping != label_mapping(metadata['id2label']):
            raise ContractError('Model label mapping differs from its receipt')
    except (KeyError, ValueError, TypeError) as exc:
        raise ContractError(f'Invalid pinned model receipt: {exc}') from exc
    return metadata


class FinBERTScorer:
    def __init__(self, settings: SentimentConfig, model_root: Path, offline: bool = True):
        self.settings = settings
        self.artifacts = prepare_model(settings, model_root, offline=offline)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ContractError('Install requirements-sentiment.lock for FinBERT inference') from exc
        self.torch = torch
        torch.set_num_threads(settings.cpu_threads)
        torch.manual_seed(settings.seed)
        torch.use_deterministic_algorithms(True)
        directory = model_root / settings.revision
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                attn_implementation='eager').to('cpu').float()
        except Exception as exc:
            raise ContractError(f'Cannot load the pinned FinBERT model ({type(exc).__name__})') from None
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        self.mapping = label_mapping(self.model.config.id2label)
        if self.mapping != label_mapping(self.artifacts['id2label']):
            raise ContractError('Loaded model labels differ from pinned metadata')
        self.model.eval()
        self.metadata = {'backend': 'finbert-pytorch-cpu', 'config': asdict(settings), 'artifacts': self.artifacts,
                         'preprocessing': PREPROCESSING_VERSION, 'padding': 'dynamic-right', 'truncation': 'right',
                         'dtype': 'float32', 'attention': 'eager',
                         'python_version': platform.python_version(), 'architecture': platform.machine(),
                         'inference_code_sha256': file_hash(Path(__file__)),
                         'dependencies': {name: version(name) for name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'huggingface-hub')}}
        self.signature = digest(self.metadata)

    def predict(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        if len(texts) > self.settings.batch_size:
            raise ContractError('FinBERT batch exceeds configured batch_size')
        texts = [normalize_text(text) for text in texts]
        original = self.tokenizer(texts, add_special_tokens=True, truncation=False, return_attention_mask=False, return_token_type_ids=False)
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.settings.max_length, return_tensors='pt')
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = self.torch.softmax(logits.float(), dim=-1).cpu().tolist()
        return [validate_prediction({**{self.mapping[index]: value for index, value in enumerate(probability)},
                    'token_count': len(tokens), 'truncated': len(tokens) > self.settings.max_length})
                for probability, tokens in zip(probabilities, original['input_ids'], strict=True)]
