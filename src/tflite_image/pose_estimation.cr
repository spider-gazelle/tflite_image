require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image"
require "./image_offset_calculations"

class TensorflowLite::Image::PoseEstimation
  include Image::Common

  enum BodyJoint
    Nose          =  0
    LeftEye       =  1
    RightEye      =  2
    LeftEar       =  3
    RightEar      =  4
    LeftShoulder  =  5
    RightShoulder =  6
    LeftElbow     =  7
    RightElbow    =  8
    LeftWrist     =  9
    RightWrist    = 10
    LeftHip       = 11
    RightHip      = 12
    LeftKnee      = 13
    RightKnee     = 14
    LeftAnkle     = 15
    RightAnkle    = 16
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(image : Canvas | FFmpeg::Frame) : Array(PoseEstimation::Output)
    apply_canvas_to_input_tensor image

    # execute the neural net
    client.invoke!

    # ensure the outputs are a float value between 0 and 1
    outputs = normalize_output_layer client.output(0)

    points = Array(PoseEstimation::Point).new(17)
    outputs.each_slice(3).each_with_index do |values, index|
      points << PoseEstimation::Point.new(BodyJoint.from_value(index), values[0], values[1], values[2])
    end

    [PoseEstimation::Output.new(points)]
  end

  # adjust the detections so they can be applied directly to the source image (or a scaled version in the same aspect ratio)
  #
  # you can run `detection_adjustments` just once and then apply them to detections for each invokation using this function
  def adjust(detections : Array(Output), target_width : Int32, target_height : Int32, offset_left : Int32, offset_top : Int32) : Array(Output)
    return detections if offset_left.zero? && offset_top.zero?
    height = target_height - offset_top - offset_top
    width = target_width - offset_left - offset_left
    detections.each(&.make_adjustment(width, height, target_width, target_height, offset_left, offset_top))
    detections
  end

  # :ditto:
  def adjust(detections : Array(Output), image : Canvas, offset_left : Int32, offset_top : Int32) : Array(Output)
    adjust(detections, image.width, image.height, offset_left, offset_top)
  end
end

require "./pose_estimation/*"
