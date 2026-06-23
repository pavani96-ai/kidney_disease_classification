from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model Pipeline"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def prepare_base_model_pipeline(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model()
            prepare_base_model = PrepareBaseModel(config =prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelPipeline()
        obj.prepare_base_model_pipeline()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)