from torch.ao.quantization import (
    QConfig,
    QConfigMapping,
    HistogramObserver,
    MovingAverageMinMaxObserver,
)
import torch

torch.backends.quantized.engine = "fbgemm"
qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(reduce_range=True),
    weight=MovingAverageMinMaxObserver.with_args(reduce_range=True),
)


qconfig_mapping = QConfigMapping().set_global(qconfig)
