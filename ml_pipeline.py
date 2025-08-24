from configurations.constants import LINE_BREAK
from services import read_dataset_service as read_dataset_service, dataset_preprocessing_service, model_training_service
from helpers.logger import logger


def main():
    try:
        logger.info(LINE_BREAK)
        logger.info("*MACHINE LEARNING WORKSPACE*")
        dataset_name = read_dataset_service.pick_sample_dataset()
        dataset, config = read_dataset_service.read_dataset(dataset_name)
        dataset_splits = dataset_preprocessing_service.preprocess_dataset(dataset, config)
        model_training_service.train_models(dataset_splits, config)
    except Exception as exception:
        logger.error(f"\tError in workflow. Exception: {str(exception)}")


if __name__ == "__main__":
    main()
