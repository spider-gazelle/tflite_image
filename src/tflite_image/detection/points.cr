require "json"
require "../detection"

module TensorflowLite::Image::Detection
  module Point
    include JSON::Serializable

    getter x : Float32
    getter y : Float32

    # adjust the detections so they can be applied directly to the source image
    def adjust(canvas_width : Int, canvas_height : Int, offset_left : Int, offset_top : Int)
      return if offset_left.zero? && offset_top.zero?

      width = canvas_width - offset_left - offset_left
      height = canvas_height - offset_top - offset_top

      make_adjustment(width, height, canvas_width, canvas_height, offset_left, offset_top)
      self
    end

    def make_adjustment(
      original_width : Int,
      original_height : Int,
      canvas_width : Int,
      canvas_height : Int,
      offset_left : Int,
      offset_top : Int
    ) : Nil
      x_px = (@x * original_width).round.to_i + offset_left
      y_px = (@y * original_height).round.to_i + offset_top

      @x = x_px.to_f32 / canvas_width
      @y = y_px.to_f32 / canvas_height
    end
  end

  class GenericPoint
    include Point

    def initialize(@x : Float32, @y : Float32)
    end

    def self.new(x : Float32, y : Float32)
      previous_def(x, y).as(Detection::Point)
    end
  end

  module Points
    abstract def points : Hash(String, Detection::Point)

    def adjust(canvas_width : Int, canvas_height : Int, offset_left : Int, offset_top : Int)
      return if offset_left.zero? && offset_top.zero?

      height = canvas_height - offset_top - offset_top
      width = canvas_width - offset_left - offset_left

      make_adjustment(width, height, canvas_width, canvas_height, offset_left, offset_top)
      self
    end

    def make_adjustment(
      original_width : Int,
      original_height : Int,
      canvas_width : Int,
      canvas_height : Int,
      offset_left : Int,
      offset_top : Int
    ) : Nil
      points.each_value &.make_adjustment(original_width, original_height, canvas_width, canvas_height, offset_left, offset_top)
    end
  end
end
