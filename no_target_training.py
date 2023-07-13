from druggen import NoTarget, NoTargetDataset, NoTargetTrainer, NoTargetTrainerConfig, NoTargetConfig



# smiles = BaseSmileDataset(root="data", filename="chembl_train.smi", huggingface_repo="HUBioDataLab/DrugGEN-chembl-smiles")

dataset = NoTargetDataset()

model_config = NoTargetConfig()

model = NoTarget(model_config)

trainer_config = NoTargetTrainerConfig()

trainer = NoTargetTrainer(model, trainer_config)



trainer.fit(dataset)

trainer.save_model()

# save dataset config # which is actually tells about the model tensors

