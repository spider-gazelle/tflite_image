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
      detections[point.name.as(String)] = point
    end

    @points = detections
  end

  getter type : Symbol = :pose

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
  def markup(image : Canvas, minimum_score : Float32 = 0.3_f32) : Canvas
    width, height = image.width, image.height

    @points.each_value do |point|
      next if point.score < minimum_score

      x = (width * point.x).round.to_i
      y = (height * point.y).round.to_i
      image.circle(x, y, 5, StumpyPNG::RGBA::WHITE, true)
    end

    LINES.map do |line|
      points = line.map { |point| @points[point.to_s] }

      previous = nil
      points.each do |p2|
        p1 = previous
        previous = p2
        next unless p1

        next if p1.score < minimum_score
        next if p2.score < minimum_score

        image.line(
          (width * p1.x).round.to_i,
          (height * p1.y).round.to_i,
          (width * p2.x).round.to_i,
          (height * p2.y).round.to_i,
          StumpyPNG::RGBA::WHITE
        )
      end
    end

    image
  end
end
