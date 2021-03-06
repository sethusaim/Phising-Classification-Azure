from phising.blob_storage_operations.blob_operations import Blob_Operation
from phising.mlflow_utils.mlflow_operations import MLFlow_Operations
from utils.logger import App_Logger
from utils.read_params import read_params


class Load_Prod_Model:
    """
    Description :   This class shall be used for loading the production model
    Written by  :   iNeuron Intelligence
    Version     :   1.2
    Revisions   :   Moved to setup to cloud
    """

    def __init__(self, num_clusters):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.num_clusters = num_clusters

        self.db_name = self.config["db_log"]["train_db_log"]

        self.model_container = self.config["container"]["model"]

        self.load_prod_model_log = self.config["train_db_log"]["load_prod_model"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.stag_model_dir = self.config["models_dir"]["stag"]

        self.exp_name = self.config["mlflow_config"]["experiment_name"]

        self.blob = Blob_Operation()

        self.mlflow_op = MLFlow_Operations(
            db_name=self.db_name, collection_name=self.load_prod_model_log
        )

    def load_production_model(self):
        """
        Method Name :   load_production_model
        Description :   This method is responsible for moving the models from the trained models dir to
                        prod models dir and stag models dir based on the metrics of the cluster

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.load_production_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.load_prod_model_log,
        )

        try:
            self.mlflow_op.set_mlflow_tracking_uri()

            exp = self.mlflow_op.get_experiment_from_mlflow(exp_name=self.exp_name)

            runs = self.mlflow_op.get_runs_from_mlflow(exp_id=exp.experiment_id)

            """
            Code Explaination: 
            num_clusters - Dynamically allocated based on the number of clusters created using elbow plot

            Here, we are trying to iterate over the number of clusters and then dynamically create the cols 
            where in the best model names can be found, and then copied to production or staging depending on
            the condition

            Eg- metrics.XGBoost1-best_score
            """
            reg_model_names = self.mlflow_op.get_mlflow_models()

            cols = [
                "metrics." + str(model) + "-best_score"
                for model in reg_model_names
                if model != "KMeans"
            ]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_info="Created cols for all registered model",
            )

            runs_cols = runs[cols].max().sort_values(ascending=False)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_info="Sorted the runs cols in descending order",
            )

            metrics_dict = runs_cols.to_dict()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_info="Converted runs cols to dict",
            )

            """ 
            Eg-output: For 3 clusters, 
            
            [
                metrics.XGBoost0-best_score,
                metrics.XGBoost1-best_score,
                metrics.XGBoost2-best_score,
                metrics.RandomForest0-best_score,
                metrics.RandomForest1-best_score,
                metrics.RandomForest2-best_score
            ] 

            Eg- runs_dataframe: I am only showing for 3 cols,actual runs dataframe will be different
                                based on the number of clusters
                
                since for every run cluster values changes, rest two cols will be left as blank,
                so only we are taking the max value of each col, which is nothing but the value of the metric
                

run_number  metrics.XGBoost0-best_score metrics.RandomForest1-best_score metrics.XGBoost1-best_score
    0                   1                       0.5
    1                                                                                   1                 
    2                                                                           
            """

            best_metrics_names = [
                max(
                    [
                        (file, metrics_dict[file])[0]
                        for file in metrics_dict
                        if str(i) in file
                    ]
                )
                for i in range(0, self.num_clusters)
            ]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_info="Got top model names based on the metrics of clusters",
            )

            ## best_metrics will store the value of metrics, but we want the names of the models,
            ## so best_metrics.index will return the name of the metric as registered in mlflow

            ## Eg. metrics.XGBoost1-best_score

            ## top_mn_lst - will store the top 3 model names

            top_mn_lst = [mn.split(".")[1].split("-")[0] for mn in best_metrics_names]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_info="Got the top model names",
            )

            results = self.mlflow_op.search_mlflow_models(order="DESC")

            ## results - This will store all the registered models in mlflow
            ## Here we are iterating through all the registered model and for every latest registered model
            ## we are checking if the model name is in the top 3 model list, if present we are putting that
            ## model into production or staging

            for res in results:
                for mv in res.latest_versions:
                    if mv.name in top_mn_lst:
                        self.mlflow_op.transition_mlflow_model(
                            model_version=mv.version,
                            stage="Production",
                            model_name=mv.name,
                            from_container_name=self.model_container,
                            to_container_name=self.model_container,
                        )

                    ## In the registered models, even kmeans model is present, so during prediction,
                    ## this model also needs to be in present in production, the code logic is present below

                    elif mv.name == "KMeans":
                        self.mlflow_op.transition_mlflow_model(
                            model_version=mv.version,
                            stage="Production",
                            model_name=mv.name,
                            from_container_name=self.model_container,
                            to_container_name=self.model_container,
                        )

                    else:
                        self.mlflow_op.transition_mlflow_model(
                            model_version=mv.version,
                            stage="Staging",
                            model_name=mv.name,
                            from_container_name=self.model_container,
                            to_container_name=self.model_container,
                        )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_info="Transitioning of models based on scores successfully done",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
            )
