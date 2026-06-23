from cnnClassifier import logger
from cnnClassifier.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.prepare_base_model_pipeline import PrepareBaseModelPipeline

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
