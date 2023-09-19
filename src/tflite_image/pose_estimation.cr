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
  def process(image : Canvas) : Tuple(Canvas, PoseEstimation::Output)
    apply_canvas_to_input_tensor image

    # execute the neural net
    client.invoke!

    # ensure the outputs are a float value between 0 and 1
    outputs = normalize_output_layer client.output(0)

    points = Array(PoseEstimation::Point).new(17)
    outputs.each_slice(3).each_with_index do |values, index|
      points << PoseEstimation::Point.new(BodyJoint.from_value(index), values[0], values[1], values[2])
    end

    {image, PoseEstimation::Output.new(points)}
  end
end

require "./pose_estimation/*"
