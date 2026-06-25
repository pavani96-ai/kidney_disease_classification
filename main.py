from cnnClassifier import logger
from cnnClassifier.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.prepare_base_model_pipeline import PrepareBaseModelPipeline
from cnnClassifier.pipeline.model_training_pipeline import ModelTrainingPipeline
from cnnClassifier.pipeline.model_Evaluation_pipeline import EvaluationPipeline
STAGE_NAME = "Data Ingestion Pipeline"

if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.data_ingestion_pipeline()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<")  
    except Exception as e:
        logger.exception(e)

STAGE_NAME = "Prepare Base Model Pipeline"


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelPipeline()
        obj.prepare_base_model_pipeline()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)


STAGE_NAME = "MODEL TRAINING PIPELINE"
if __name__ == "__main__":
    try:
        logger.info(f"*****************************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
    
STAGE_NAME = "Evaluation Stage"
if __name__ == "__main__":
    try:
      logger.info(f"*************************")
      logger.info(f" >>>>>> stage {STAGE_NAME} started <<<<<")
      obj = EvaluationPipeline()
      obj.main()
      logger.info(f">>>> stage {STAGE_NAME} completed <<<<\\N\\NX=======X")
    
    except Exception as e:
       logger.exception(e)
       raise e