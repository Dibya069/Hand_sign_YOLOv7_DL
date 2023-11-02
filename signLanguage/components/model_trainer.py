import os,sys
import yaml
from signLanguage.utils.main_utils import read_yaml_file
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import ModelTrainerConfig
from signLanguage.entity.artifact_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    
    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip Hand_gasture.zip")
            os.system("rm Hand_gasture.zip")

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov7/cfg/deploy/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov7/cfg/deploy/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            os.system(f"cd yolov7/ && python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./cfg/deploy/custom_yolov7x.yaml --weights {self.model_trainer_config.weight_name} --name yolov7x_results  --cache")
            os.system("cp yolov7/runs/train/yolov7x_results/weights/best.pt yolov7/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp yolov7/runs/train/yolov7x_results/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
           
            os.system("rm -rf yolov7/runs")
            os.system("rm -rf train")
            os.system("rm -rf test")
            os.system("rm -rf valid")
            os.system("rm -rf data.yaml")
            os.system("rm -rf README.dataset.txt")
            os.system("rm -rf README.roboflow.txt")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov7/runs/train/weights/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise SignException(e, sys)





    
    