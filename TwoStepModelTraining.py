from TwoStepModelTraining_step_one import train_and_save_model_binary
from TwoStepModelTraining_step_two import train_and_save_model_classification


def main():
    """
    Main function to call the train_defect_model functions.
    """

    DATADIR = "dataCollection/Data/Perfect_Data"
    model_name_binary = "Model_20250121_binary.keras"
    model_name_classification = "Model_20250121_classification.keras"
    train_and_save_model_binary(DATADIR=DATADIR, model_name=model_name_binary)
    train_and_save_model_classification(DATADIR=DATADIR, model_name=model_name_classification)

if __name__ == "__main__":
    main()