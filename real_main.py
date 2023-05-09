from druggen import DrugGENConfig, DrugGEN
from druggen import DrugGENTrainer
from druggen import DrugGENDataModule


data_module = DrugGENDataModule()

config = DrugGENConfig()
model = DrugGEN(config)

trainer = DrugGENTrainer(
    log_folder="logs",
    checkpoint_folder="checkpoints",
    wandb=False,
    rdkit_logging="ignore",
    gpus=1,
    max_epochs=1,
    
)

trainer.fit(model, data_module)

trainer.test(model, data_module)

trainer.upload_huggingface_hub("druggen", "druggen", "main")