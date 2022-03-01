import pandas as pd
from botocore.exceptions import ClientError
from phising.data_ingestion.data_loader_prediction import data_getter_pred
from phising.data_preprocessing.preprocessing import preprocessor
from phising.blob_storage_operations.blob_operations import Blob_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class prediction:
    """
    Description :   This class shall be used for loading the production model

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.config = read_params()

        self.pred_log = self.config["pred_db_log"]["pred_main"]

        self.model_container = self.config["container"]["phising_model_container"]

        self.input_files_container = self.config["container"]["inputs_files_container"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.pred_output_file = self.config["pred_output_file"]

        self.log_writer = App_Logger()

        self.blob = Blob_Operation()

        self.data_getter_pred = data_getter_pred(collection_name=self.pred_log)

        self.preprocessor = preprocessor(collection_name=self.pred_log)

        self.class_name = self.__class__.__name__

    def delete_pred_file(self, collection_name):
        """
        Method Name :   delete_pred_file
        Description :   This method is used for deleting the existing prediction batch file

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_pred_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            collection_name=collection_name,
        )

        try:
            self.blob.load_object(
                container_name=self.input_files_container, obj=self.pred_output_file
            )

            self.log_writer.log(
                collection_name=collection_name,
                log_info=f"Found existing prediction batch file. Deleting it.",
            )

            self.blob.delete_file(
                container_name=self.input_files_container,
                file=self.pred_output_file,
                collection_name=collection_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                collection_name=collection_name,
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pass

            else:
                self.log_writer.exception_log(
                    error=e,
                    class_name=self.class_name,
                    method_name=method_name,
                    collection_name=collection_name,
                )

    def find_correct_model_file(self, cluster_number, container_name, collection_name):
        """
        Method Name :   find_correct_model_file
        Description :   This method is used for finding the correct model file during prediction

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.find_correct_model_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            collection_name=collection_name,
        )

        try:
            list_of_files = self.blob.get_files(
                container=container_name,
                folder_name=self.prod_model_dir,
                collection_name=collection_name,
            )

            for file in list_of_files:
                try:
                    if file.index(str(cluster_number)) != -1:
                        model_name = file

                except:
                    continue

            model_name = model_name.split(".")[0]

            self.log_writer.log(
                collection_name=collection_name,
                log_info=f"Got {model_name} from {self.prod_model_dir} folder in {container_name} container",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                collection_name=collection_name,
            )

            return model_name

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                collection_name=collection_name,
            )

    def predict_from_model(self):
        """
        Method Name :   predict_from_model
        Description :   This method is used for loading from prod model dir of blob container and use them for prediction

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.predict_from_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            collection_name=self.pred_log,
        )

        try:
            self.delete_pred_file(collection_name=self.pred_log)

            data = self.data_getter_pred.get_data()

            is_null_present = self.preprocessor.is_null_present(data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data)

            cols_to_drop = self.preprocessor.get_columns_with_zero_std_deviation(data)

            data = self.preprocessor.remove_columns(data, cols_to_drop)

            kmeans = self.blob.load_model(
                container=self.model_container,
                model_name="KMeans",
                collection_name=self.pred_log,
            )

            clusters = kmeans.predict(data.drop(["phising"], axis=1))

            data["clusters"] = clusters

            clusters = data["clusters"].unique()

            for i in clusters:
                cluster_data = data[data["clusters"] == i]

                phising_names = list(cluster_data["phising"])

                cluster_data = data.drop(labels=["phising"], axis=1)

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                crt_model_name = self.find_correct_model_file(
                    cluster_number=i,
                    container_name=self.model_container,
                    collection_name=self.pred_log,
                )

                model = self.blob.load_model(model_name=crt_model_name)

                result = list(model.predict(cluster_data))

                result = pd.DataFrame(
                    list(zip(phising_names, result)), columns=["phising", "Prediction"]
                )

                self.blob.upload_df_as_csv(
                    data_frame=result,
                    file_name=self.pred_output_file,
                    container=self.input_files_container,
                    dest_file_name=self.pred_output_file,
                    collection_name=self.pred_log,
                )

            self.log_writer.log(
                collection_name=self.pred_log, log_info=f"End of Prediction"
            )

            return (
                self.input_files_container,
                self.pred_output_file,
                result.head().to_json(orient="records"),
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                collection_name=self.pred_log,
            )
