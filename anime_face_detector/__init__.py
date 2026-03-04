from .detector_onnx import (
    LandmarkDetectorONNX,
    create_detector_onnx,
    get_onnx_paths,
)

__all__ = [
    "LandmarkDetectorONNX",
    "create_detector",
    "create_detector_onnx",
    "get_onnx_paths",
]


def create_detector(
    face_detector_name: str = "yolov3",
    landmark_model_name: str = "hrnetv2",
    device: str = "cpu",
    box_scale_factor: float = 1.1,
    require_cuda: bool = False,
) -> LandmarkDetectorONNX:
    """
    在线服务仅使用 ONNX 推理，默认要求 onnx_models/ 下已放置导出的模型。
    """
    assert face_detector_name in ["yolov3"]
    assert landmark_model_name in ["hrnetv2"]
    face_onnx, landmark_onnx = get_onnx_paths(face_detector_name)
    return create_detector_onnx(
        face_name=face_detector_name,
        face_onnx_path=face_onnx,
        landmark_onnx_path=landmark_onnx,
        device=device,
        box_scale_factor=box_scale_factor,
        require_cuda=require_cuda,
    )
