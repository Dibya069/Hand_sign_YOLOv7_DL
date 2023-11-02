from signLanguage.pipeline.training_pipeline import TrainPipeline

obj = TrainPipeline()
data_ingestion = obj.start_data_ingestion()
data_validation = obj.start_data_validation(data_ingestion)

if data_validation.validation_status == True:
    model_train = obj.start_model_trainer()