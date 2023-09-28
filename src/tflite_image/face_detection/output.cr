require "../face_detection"

class TensorflowLite::Image::FaceDetection::Output
  include Detection
  include Detection::Points
  include Detection::BoundingBox

  LABEL = "face"

  def initialize(
    @top : Float32,
    @left : Float32,
    @bottom : Float32,
    @right : Float32,
    @score : Float32,
    @index : Int32,
    points : Array(FaceDetection::Point)
  )
    @label = LABEL

    detections = {} of String => FaceDetection::Point
    points.each do |point|
      detections[point.label.as(String)] = point
    end

    @points = detections
  end

  getter type : Symbol = :face

  def boundary : Detection::Line
    points = [
      Detection::GenericPoint.new(@left, @top),
      Detection::GenericPoint.new(@right, @top),
      Detection::GenericPoint.new(@right, @bottom),
      Detection::GenericPoint.new(@left, @bottom),
    ]
    Detection::Line.new(points)
  end

  def lines : Indexable(Detection::Line)
    {boundary}
  end

  @points : Hash(String, FaceDetection::Point)

  def points : Hash(String, Detection::Point)
    @points.transform_values &.as(Detection::Point)
  end

  def area : Float32
    (@right - @left) * (@top - @bottom)
  end

  # adjust the detections so they can be applied directly to the source image
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
    top_px = (top * original_height).round.to_i + offset_top
    bottom_px = (bottom * original_height).round.to_i + offset_top
    left_px = (left * original_width).round.to_i + offset_left
    right_px = (right * original_width).round.to_i + offset_left

    @left = left_px.to_f32 / canvas_width
    @right = right_px.to_f32 / canvas_width
    @bottom = bottom_px.to_f32 / canvas_height
    @top = top_px.to_f32 / canvas_height

    points.each_value &.make_adjustment(original_width, original_height, canvas_width, canvas_height, offset_left, offset_top)
  end

  def markup(image : Canvas, minimum_score : Float32 = 0.3_f32, font : PCFParser::Font? = nil) : Canvas
    width, height = image.width, image.height

    points = boundary.points
    points.each_with_index do |point, index|
      next_point = points[(index + 1) % points.size]
      image.line(
        (point.x * width).round.to_i,
        (point.y * height).round.to_i,
        (next_point.x * width).round.to_i,
        (next_point.y * height).round.to_i,
        StumpyPNG::RGBA::WHITE
      )
    end

    @points.each_value do |point|
      x = (width * point.x).round.to_i
      y = (height * point.y).round.to_i
      image.circle(x, y, 5, StumpyPNG::RGBA::WHITE, true)
    end

    image
  end
end
