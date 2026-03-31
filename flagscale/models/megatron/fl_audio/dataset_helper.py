from dataclasses import dataclass
import dataclasses
import torch
from typing import Optional
from megatron.energon import (
    DefaultTaskEncoder,
    Batch,
    Cooker
)
from megatron.energon.task_encoder.base import stateless

@dataclass
class FLAudioSample:
    __key__: str
    __restore_key__: str
    audio_ids:  torch.tensor

@dataclass
class FLAudioSampleBatch(Batch):
    __key__: str
    __restore_key__: str
    audio_ids:  torch.tensor
@stateless
def cook_text(sample: dict) -> FLAudioSample:
    key = int(sample["__key__"].strip('/').split('/')[-1])
    return FLAudioSample(
        __key__   = key,
        audio_ids = sample["audio_ids.pyd"],
        __restore_key__ = sample["__restore_key__"]
    )

class FLAudioTaskEncoder(DefaultTaskEncoder[FLAudioSample, FLAudioSample,FLAudioSampleBatch, dict]):
    cookers = [Cooker(cook_text)]

    @stateless
    def encode_sample(self, sample: dict) -> FLAudioSample:
        
        return FLAudioSample(
            __key__   = sample.__key__,
            __restore_key__ = sample.__restore_key__,
            audio_ids = sample.audio_ids,
        )

    def batch(self, samples: list[FLAudioSample]) -> FLAudioSampleBatch:

        input_batch = torch.cat([s.audio_ids for s in samples], dim=0)
        return FLAudioSampleBatch(
            __key__   = [s.__key__ for s in samples],
            __restore_key__ = [s.__restore_key__ for s in samples],
            audio_ids = input_batch,
        )

    def encode_batch(self, batch: FLAudioSampleBatch) -> dict:
        raw = dataclasses.asdict(batch)
        return raw

def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
