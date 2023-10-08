from pytorch_lightning import Trainer

from loa.datamodule import LOADatamodule
from loa.lit_loa import LitLOA

ckp_path = ""

if __name__ == "__main__":
    trainer = Trainer(logger=False, accelerator="gpu", devices=1)

    dm = LOADatamodule(data_type="ours")

    model = LitLOA.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)
