import yaml


def read_params(config_path="params.yaml"):
    """
    Method Name :   read_params
    Description :   This method is used for read the params from yaml file

    Version     :   1.2
    Revisions   :   Moved to setup to cloud
    """
    method_name = read_params.__name__

    local_file_name = __file__

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise Exception(
            f"Exception occured in {local_file_name}, Method : {method_name}, Error : {str(e)}"
        )