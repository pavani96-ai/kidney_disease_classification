from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig
class ConfigurationManager:
    def __init__(self,
                 config_filepath =CONFIG_FILE_PATH,
                 params_filepath =PARAMS_FILE_PATH,
                 schema_filepath =SCHEMA_FILE_PATH
                 ):
        self.config =read_yaml(config_filepath)
        self.params =read_yaml(params_filepath)
        self.schema =read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        root_dir = Path(config.root_dir)
        source_url = config.source_URL
        local_data_file = Path(config.local_data_file)
        unzip_dir = Path(config.unzip_dir)

        create_directories([config.root_dir,local_data_file.parent,unzip_dir.parent])

        data_ingestion_config = DataIngestionConfig(
            root_dir = root_dir,
            source_URL =source_url,
            local_data_file = local_data_file,
            unzip_dir = unzip_dir )
        
        return data_ingestion_config
    
    def get_prepare_base_model(self):
        config = self.config.prepare_base_model

        root_dir = Path(config.root_dir)
        base_model_path = Path(config.base_model_path)
        updated_base_model_path = Path(config.updated_base_model_path)

        create_directories([config.root_dir,base_model_path.parent,updated_base_model_path.parent])


        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = root_dir,
            base_model_path = base_model_path,
            updated_base_model_path = updated_base_model_path,
            params_image_size = list(self.params.IMAGE_SIZE),
            params_learning_rate=float(self.params.LEARNING_RATE),
            params_include_top = bool(self.params.INCLUDE_TOP),
            params_weights = str(self.params.WEIGHTS),
            params_classes =int(self.params.CLASSES),
            
)
        return prepare_base_model_config
