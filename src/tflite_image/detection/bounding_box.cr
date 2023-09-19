require "./classification"
require "./lines"

module TensorflowLite::Image::Detection::BoundingBox
  include Classification
  include Lines

  abstract def boundary : Indexable(Lines)
end
