"""
Utility helper functions for execution provider detection and system checks.
"""

def get_available_providers():
    """
    Detects available ONNX Runtime execution providers.
    Prioritizes CUDA and DirectML before falling back to CPU execution.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers
    except ImportError:
        return ['CPUExecutionProvider']
