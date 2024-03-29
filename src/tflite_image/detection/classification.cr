require "../detection"

module TensorflowLite::Image::Detection::Classification
  getter index : Int32
  getter score : Float32
  getter label : String? = nil
end
