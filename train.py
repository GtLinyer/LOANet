from pytorch_lightning.utilities.cli import LightningCLI

from loa.datamodule import LOADatamodule
from loa.lit_loa import LitLOA

cli = LightningCLI(LitLOA, LOADatamodule)

# python train.py fit -c config.yaml
