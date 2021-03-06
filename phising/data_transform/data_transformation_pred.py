from phising.blob_storage_operations.blob_operations import Blob_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Data_Transform_Pred:
    """
    Description :  This class shall be used for transforming the prediction batch data before loading it in Database!!.

    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.pred_data_container = self.config["container"]["phising_pred_data"]

        self.blob = Blob_Operation()

        self.log_writer = App_Logger()

        self.good_pred_data_dir = self.config["data"]["pred"]["good_data_dir"]

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_data_transform_log = self.config["pred_db_log"]["data_transform"]

    def add_quotes_to_string(self):
        """
        Method Name :   add_quotes_to_string
        Description :   This method addes the quotes to the string data present in columns

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.add_quotes_to_string.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            collection_name=self.pred_data_transform_log,
        )

        try:
            lst = self.blob.read_csv_from_folder(
                folder_name=self.good_pred_data_dir,
                container_name=self.pred_data_container,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

            for f in lst:
                df = f[0]

                file = f[1]

                abs_f = f[2]

                if file.endswith(".csv"):
                    for column in df.columns:
                        count = df[column][df[column] == "?"].count()

                        if count != 0:
                            df[column] = df[column].replace("?", "'?'")

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                        log_info=f"Quotes added for the file {file}",
                    )

                    self.blob.upload_df_as_csv(
                        data_frame=df,
                        local_file_name=abs_f,
                        container_file_name=file,
                        container=self.pred_data_container,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )
