from druggen import DrugGENConfig, DrugGEN
from druggen import DrugGENTrainer
from druggen import DrugGENDataModule


data_module = DrugGENDataModule()

config = DrugGENConfig()
model = DrugGEN(config)

trainer = DrugGENTrainer(
    model=model,
    model_config=config,
    log_folder="logs",
    checkpoint_folder="checkpoints",
    wandb=False,
    disable_rdkit_logging=True,
    max_epochs=1,
    
)

trainer.fit(data_module)

trainer.test(data_module)

trainer.upload_huggingface_hub