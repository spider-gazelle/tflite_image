require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image.cr"
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

  class Detection
    include JSON::Serializable

    def initialize(@joint, @y, @x, @score)
    end

    getter joint : BodyJoint
    getter score : Float32
    getter x : Float32
    getter y : Float32

    getter! x_px : Int32
    getter! y_px : Int32

    def position(width, height, offset_left = 0, offset_top = 0)
      height = height - offset_top - offset_top
      width = width - offset_left - offset_left

      @y_px = y_px = (y * height).round.to_i + offset_top
      @x_px = x_px = (x * width).round.to_i + offset_left

      {x_px, y_px}
    end
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(canvas : Canvas) : Tuple(Canvas, Hash(BodyJoint, Detection))
    apply_canvas_to_input_tensor canvas

    # execute the neural net
    client.invoke!

    # ensure the outputs are a float value between 0 and 1
    outputs = normalize_output_layer client.output(0)

    detections = {} of BodyJoint => Detection
    outputs.each_slice(3).each_with_index do |values, index|
      detection = Detection.new(BodyJoint.from_value(index), values[0], values[1], values[2])
      detections[detection.joint] = detection
    end

    {canvas, detections}
  end

  # The skeleton we want to draw
  LINES = [
    [BodyJoint::LeftEar, BodyJoint::LeftEye, BodyJoint::Nose, BodyJoint::RightEye, BodyJoint::RightEar],
    [
      BodyJoint::LeftWrist, BodyJoint::LeftElbow, BodyJoint::LeftShoulder,
      BodyJoint::RightShoulder, BodyJoint::RightElbow, BodyJoint::RightWrist,
    ],
    [BodyJoint::LeftShoulder, BodyJoint::LeftHip, BodyJoint::LeftKnee, BodyJoint::LeftAnkle],
    [BodyJoint::RightShoulder, BodyJoint::RightHip, BodyJoint::RightKnee, BodyJoint::RightAnkle],
    [BodyJoint::LeftHip, BodyJoint::RightHip],
  ]

  # add the detection details to an image
  #
  # if marking up the original image,
  # you'll need to take into account how it was scaled and provide offsets
  def markup(canvas : Canvas, detections : Hash(BodyJoint, Detection), left_offset : Int32 = 0, top_offset : Int32 = 0, minimum_score : Float32 = 0.5_f32) : Canvas
    detections.each_value do |detection|
      next if detection.score < minimum_score

      x, y = detection.position(canvas.width, canvas.height, left_offset, top_offset)
      canvas.circle(x, y, 5, StumpyPNG::RGBA::WHITE, true)
    end

    LINES.each do |points|
      points.each_cons(2, reuse: true) do |line|
        p1 = detections[line[0]]
        next if p1.score < minimum_score
        p2 = detections[line[1]]
        next if p2.score < minimum_score

        canvas.line(p1.x_px, p1.y_px, p2.x_px, p2.y_px, StumpyPNG::RGBA::WHITE)
      end
    end

    canvas
  end
end
