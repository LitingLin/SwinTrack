from typing import Optional, List


class BaseRunner:
    def get_iteration_index(self) -> Optional[int]:
        raise NotImplementedError

    def register_data_pipelines(self, branch_name: str, data_pipelines: dict):
        raise NotImplementedError

    def get_metric_definitions(self) -> List[dict]:
        raise NotImplementedError

    def switch_branch(self, branch_name: str):
        raise NotImplementedError

    def train(self, is_train: bool):
        raise NotImplementedError

    def run_iteration(self, model, data):
        pass
