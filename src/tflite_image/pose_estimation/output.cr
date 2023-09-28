require "../pose_estimation"
require "./point"

class TensorflowLite::Image::PoseEstimation::Output
  include Detection
  include Detection::Lines
  include Detection::Points

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

  def initialize(points : Array(PoseEstimation::Point))
    detections = {} of String => PoseEstimation::Point
    points.each do |point|
      detections[point.label.as(String)] = point
    end

    @points = detections
  end

  getter type : Symbol = :pose

  def label : String?
    nil
  end

  def lines : Indexable(Detection::Line)
    LINES.map do |line|
      line = line.map { |point| @points[point.to_s].as(Detection::Point) }
      Detection::Line.new(line)
    end
  end

  @points : Hash(String, PoseEstimation::Point)

  def points : Hash(String, Detection::Point)
    @points.transform_values &.as(Detection::Point)
  end

  # add the detection details to an image
  #
  # if marking up the original image,
  # you'll need to take into account how it was scaled and provide offsets
  def markup(image : Canvas, minimum_score : Float32 = 0.3_f32, font : PCFParser::Font? = nil) : Canvas
    width, height = image.width, image.height

    @points.each_value do |point|
      next if point.score < minimum_score

      x = (width * point.x).round.to_i
      y = (height * point.y).round.to_i
      image.circle(x, y, 5, StumpyPNG::RGBA::WHITE, true)
    end

    LINES.map do |line|
      points = line.map { |point| @points[point.to_s] }

      points.each_with_index do |point, index|
        next_point = points[index + 1]?
        next unless next_point
        image.line(
          (point.x * width).round.to_i,
          (point.y * height).round.to_i,
          (next_point.x * width).round.to_i,
          (next_point.y * height).round.to_i,
          StumpyPNG::RGBA::WHITE
        )
      end
    end

    image
  end
end
