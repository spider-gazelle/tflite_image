require "../face_detection"

class TensorflowLite::Image::FaceDetection::Point
  include Detection::Point
  include Detection::Classification

  getter feature : FaceFeatures

  def initialize(@feature, @x, @y, @score)
    @index = @feature.value
    @label = @feature.to_s
  end
end
