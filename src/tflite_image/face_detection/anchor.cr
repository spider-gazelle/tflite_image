require "../face_detection"

struct TensorflowLite::Image::FaceDetection::Anchor
  def initialize(@x_center, @y_center)
  end

  property x_center : Float32
  property y_center : Float32
end
