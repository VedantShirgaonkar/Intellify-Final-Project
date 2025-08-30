class ModelHandler:
    def log_model_architecture(self, config: dict):
        pass
    def log_parameters_count(self, total: int, trainable: int):
        pass
    def log_model_loading(self, path: str, success: bool):
        pass

class DetectionHandler:
    def log_detections(self, detections, frame_id=None):
        pass
    def log_inference_time(self, ms, fps):
        pass

def create_detection_live_display():
    pass
