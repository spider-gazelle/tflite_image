require "json"
require "./detection/*"

module TensorflowLite::Image::Detection
  include JSON::Serializable

  # i.e. object, pose, face, age, class
  abstract def type : Symbol

  getter associated : Array(Detection) { [] of Detection }

  # :nodoc:
  def adjust(canvas_width : Int, canvas_height : Int, offset_left : Int, offset_top : Int)
    self
  end

  # :nodoc:
  def adjust(image : Canvas, offset_left : Int, offset_top : Int)
    self
  end

  # :nodoc:
  def make_adjustment(
    original_width : Int,
    original_height : Int,
    canvas_width : Int,
    canvas_height : Int,
    offset_left : Int,
    offset_top : Int
  ) : Nil
  end
end
