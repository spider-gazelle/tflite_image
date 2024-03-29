require "json"
require "yaml"
require "./detection/*"

module TensorflowLite::Image::Detection
  include JSON::Serializable
  include YAML::Serializable

  # i.e. object, pose, face, age, class
  abstract def type : Symbol
  abstract def label : String?

  property associated : Array(Detection) { [] of Detection }

  # :nodoc:
  def adjust(canvas_width : Int, canvas_height : Int, offset_left : Int, offset_top : Int)
    self
  end

  # :nodoc:
  def adjust(image, offset_left : Int, offset_top : Int)
    adjust(image.width, image.height, offset_left, offset_top)
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

  def markup(image : Canvas, minimum_score : Float32 = 0.3_f32, font : PCFParser::Font? = nil, color = StumpyPNG::RGBA::WHITE) : Canvas
    image
  end
end
