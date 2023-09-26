import typer
from .workflows import LDMTrainer
import yaml

app = typer.Typer()


@app.command()
def train(config: str, vae_config: str, vae_ckpt: str):
    config = yaml.full_load(open(config))
    vae_config = yaml.full_load(open(vae_config))
    LDMTrainer(config, vae_config, vae_ckpt).start()


app()
