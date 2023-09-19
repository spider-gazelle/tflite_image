require "../pose_estimation"

class TensorflowLite::Image::PoseEstimation::Point
  include Detection::Point
  include Detection::Classification

  getter joint : BodyJoint

  def initialize(@joint, @y, @x, @score)
    @index = @joint.value
    @name = @joint.to_s
  end
end
