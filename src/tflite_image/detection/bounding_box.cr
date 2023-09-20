require "./classification"
require "./lines"

module TensorflowLite::Image::Detection::BoundingBox
  include Classification
  include Lines

  getter top : Float32
  getter left : Float32
  getter bottom : Float32
  getter right : Float32

  abstract def boundary : Detection::Line
end
