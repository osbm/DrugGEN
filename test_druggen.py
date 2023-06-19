import druggen


dataset = druggen.datasets.DrugGENNoTargetDataset(
    data_folder="data2",
)

model_config = druggen.models.DrugGENNoTargetConfig()

model = druggen.models.DrugGENNoTarget(config=model_config)

trainer_config = druggen.models.DrugGENNoTargetTrainerConfig()

trainer = druggen.models.DrugGENNoTargetTrainer(config = trainer_config, model = model)


trainer.train(dataset)

trainer.save_checkpoint("checkpoint")
