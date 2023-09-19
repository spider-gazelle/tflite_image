require "json"
require "../detection"

module TensorflowLite::Image::Detection
  class Line
    include JSON::Serializable

    def initialize(@points : Array(Detection::Point))
    end

    getter points : Array(Detection::Point)
  end

  module Lines
    abstract def lines : Indexable(Detection::Line)
  end
end
