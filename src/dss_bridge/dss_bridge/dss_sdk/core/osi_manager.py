import numpy as np

from google.protobuf.message import DecodeError
from google.protobuf.json_format import MessageToJson

from ..osi3 import __version__ as osi_version
from ..osi3 import osi_sensorview_pb2, osi_groundtruth_pb2

from ..utils.logger import logger

class OSIManager:
    @staticmethod
    def get_osi_version():
        """
        Simple function to output the OSI version and Protocol Buffers version
        
        Returns:
            str: OSI and Protocol Buffers version information
        """
        result = []
        
        # Get OSI version
        try:
            result.append(f"ASAM OSI version: {osi_version}")
        except ImportError:
            result.append("Unable to retrieve ASAM OSI version information.")
        
        # Get Protocol Buffers version
        try:
            import google.protobuf
            result.append(f"Protocol Buffers Version: {google.protobuf.__version__}")
        except (ImportError, AttributeError):
            result.append("Unable to retrieve Protocol Buffers version information.")
        
        return ", ".join(result)
    

    @staticmethod
    def parse_ground_truth(binary_data: bytes):
        if not binary_data:
            logger.warning("No ground truth binary data")
            return None
            
        try:
            ground_truth = osi_groundtruth_pb2.GroundTruth()
            ground_truth.ParseFromString(binary_data)
            return ground_truth
        except DecodeError as e:
            logger.error(f"Failed to parse GroundTruth: {e}")
            return None


