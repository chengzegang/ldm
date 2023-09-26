from torch.ao.quantization import (
    QConfig,
    QConfigMapping,
    HistogramObserver,
    get_default_qconfig,
    default_histogram_observer,
)
import torch

torch.backends.quantized.engine = "fbgemm"
default_qconfig = get_default_qconfig("fbgemm")
qconfig_conv_transpose = QConfig(
    activation=HistogramObserver.with_args(reduce_range=True),
    weight=default_histogram_observer,
)


qconfig_mapping = (
    QConfigMapping()
    .set_object_type("LayerNorm", default_qconfig)
    .set_object_type("Linear", default_qconfig)
    .set_object_type("SiLU", default_qconfig)
    .set_object_type("InstanceNorm", default_qconfig)
    .set_object_type("Conv2d", default_qconfig)
    .set_object_type("ConvTranspose2d", qconfig_conv_transpose)
    .set_object_type("MultiheadAttention", default_qconfig)
)
