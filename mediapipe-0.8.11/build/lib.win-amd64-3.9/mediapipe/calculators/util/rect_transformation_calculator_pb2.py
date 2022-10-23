# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/rect_transformation_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n?mediapipe/calculators/util/rect_transformation_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xa4\x02\n#RectTransformationCalculatorOptions\x12\x12\n\x07scale_x\x18\x01 \x01(\x02:\x01\x31\x12\x12\n\x07scale_y\x18\x02 \x01(\x02:\x01\x31\x12\x10\n\x08rotation\x18\x03 \x01(\x02\x12\x18\n\x10rotation_degrees\x18\x04 \x01(\x05\x12\x0f\n\x07shift_x\x18\x05 \x01(\x02\x12\x0f\n\x07shift_y\x18\x06 \x01(\x02\x12\x13\n\x0bsquare_long\x18\x07 \x01(\x08\x12\x14\n\x0csquare_short\x18\x08 \x01(\x08\x32\\\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x88\x83\x85} \x01(\x0b\x32..mediapipe.RectTransformationCalculatorOptions')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.rect_transformation_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_RECTTRANSFORMATIONCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  _RECTTRANSFORMATIONCALCULATOROPTIONS._serialized_start=117
  _RECTTRANSFORMATIONCALCULATOROPTIONS._serialized_end=409
# @@protoc_insertion_point(module_scope)
