require "tensorflow_lite"
require "stumpy_core"
require "stumpy_utils"
require "stumpy_resize"
require "math"
require "../tflite_image.cr"
require "./image_offset_calculations"

class TensorflowLite::Image::FaceDetection
  include Image::Common

  struct Anchor
    def initialize(@x_center, @y_center)
    end

    property x_center : Float32
    property y_center : Float32
  end

  getter anchors : Array(Anchor) = [] of Anchor
  property confidence_threshold : Float32 = 0.6_f32
  property nms_similarity_threshold : Float32 = 0.5_f32

  # top, left, bottom, right are all percentages
  record Detection,
    top : Float32,
    left : Float32,
    bottom : Float32,
    right : Float32,
    points : Array(Tuple(Float32, Float32)),
    index : Int32,
    score : Float64 do
    include JSON::Serializable

    @[JSON::Field(ignore: true)]
    @adjusted = false

    def lines(width, height, offset_left = 0, offset_top = 0)
      adjust(width, height, offset_left, offset_top)

      top_px = (top * height).round.to_i
      bottom_px = (bottom * height).round.to_i
      left_px = (left * width).round.to_i
      right_px = (right * width).round.to_i

      {
        # top line
        {left_px, top_px, right_px, top_px},
        # left line
        {left_px, top_px, left_px, bottom_px},
        # right line
        {right_px, top_px, right_px, bottom_px},
        # bottom line
        {left_px, bottom_px, right_px, bottom_px},
      }
    end

    def adjust(canvas_width, canvas_height, offset_left, offset_top)
      return if @adjusted || offset_left == 0 && offset_top == 0

      height = canvas_height - offset_top - offset_top
      width = canvas_width - offset_left - offset_left

      top_px = (top * height).round.to_i + offset_top
      bottom_px = (bottom * height).round.to_i + offset_top
      left_px = (left * width).round.to_i + offset_left
      right_px = (right * width).round.to_i + offset_left

      @left = left_px.to_f32 / canvas_width
      @right = right_px.to_f32 / canvas_width
      @bottom = bottom_px.to_f32 / canvas_height
      @top = top_px.to_f32 / canvas_height

      @points.map! do |(x, y)|
        x_px = (x * height).round.to_i + offset_left
        y_px = (y * width).round.to_i + offset_top
        {
          x_px.to_f32 / canvas_width,
          y_px.to_f32 / canvas_height,
        }
      end
    ensure
      @adjusted = true
    end

    def points(width, height, offset_left = 0, offset_top = 0)
      adjust(width, height, offset_left, offset_top)
      @points.map do |xy|
        {
          (xy[0] * width).round.to_i,
          (xy[1] * height).round.to_i,
        }
      end
    end

    def area : Float32
      (@right - @left) * (@top - @bottom)
    end
  end

  # attempts to classify the object, assumes the image has already been prepared
  def process(image : Canvas) : Tuple(Canvas, Array(Detection))
    apply_canvas_to_input_tensor image

    # execute the neural net
    client.invoke!

    # https://www.tensorflow.org/lite/examples/object_detection/overview#output_signature
    # ensure the outputs are a float value between 0 and 1
    # collate the results (convert bounding boxes from pixels to percentages)
    outputs = client.outputs

    # These values are the bounding box parameters and the facial keypoints'
    # relative coordinates to the corresponding anchor
    bounding_boxes = normalize_output_layer(outputs[0])

    # how many features are we tracking for the bounding boxes i.e. 1x896x[16]
    num_points = outputs[0].dimension_size(2)

    # represents the probability of the presence of a face at each of the anchor points
    anchor_probability = sigmoid normalize_output_layer(outputs[1])
    raise "probability and anchor size mismatch (#{anchor_probability.size} != #{@anchors.size})" unless anchor_probability.size == @anchors.size

    # width == height so scale is the same across the board
    scale = resolution[0].to_f32

    # transform the results and sort by confidence
    detections = [] of Detection
    anchor_probability.each_with_index do |probability, index|
      next unless probability >= @confidence_threshold

      # return a sub-slice representing the box at the index
      idx = index * num_points
      box = bounding_boxes[idx, num_points]

      # ignore results with negative width or height
      next if box[2].negative? || box[3].negative?

      # scale the values to percentages
      anchor = @anchors[index]
      anchor_x = anchor.x_center
      anchor_y = anchor.y_center
      box.each_index { |i| box[i] = box[i] / scale }

      # adjust the center pixel
      dx, dy, dw, dh = box[0], box[1], box[2], box[3]
      x_center = anchor_x + dx
      y_center = anchor_y + dy

      # Calculate the bounding box values
      half_width = dw / 2
      half_height = dh / 2
      xmin = x_center - half_width
      xmax = x_center + half_width
      ymin = y_center - half_height
      ymax = y_center + half_height

      points = box[4..-1].each_slice(2).to_a.map { |slice| {anchor_x + slice[0], anchor_y + slice[1]} }

      detections << Detection.new(
        top: ymax,
        left: xmin,
        bottom: ymin,
        right: xmax,
        points: points,
        index: index,
        score: probability
      )
    end

    {image, non_maximum_suppression(detections, @nms_similarity_threshold)}
  end

  # adjust the detections so they can be applied directly to the source image (or a scaled version in the same aspect ratio)
  #
  # you can run `detection_adjustments` just once and then apply them to detections for each invokation using this function
  def adjust(target_width : Int32, target_height : Int32, detections : Array(Detection), left_offset : Int32, top_offset : Int32) : Array(Detection)
    detections.each(&.adjust(target_width, target_height, left_offset, top_offset))
  end

  # :ditto:
  def adjust(image : Canvas, detections : Array(Detection), left_offset : Int32, top_offset : Int32) : Array(Detection)
    adjust(image.width, image.height, detections, left_offset, top_offset)
  end

  # add the detection details to an image
  #
  # if marking up the original image,
  # you'll need to take into account how it was scaled
  def markup(image : Canvas, detections : Array(Detection), left_offset : Int32 = 0, top_offset : Int32 = 0, minimum_score : Float32 = 0.3_f32, font : PCFParser::Font? = nil) : Canvas
    detections.each do |detection|
      next if detection.score < minimum_score

      lines = detection.lines(image.width, image.height, left_offset, top_offset)
      lines.each do |line|
        image.line(*line)
      end
      detection.points(image.width, image.height, left_offset, top_offset).each do |point|
        image.circle(*point, 4, filled: true)
      end
    end
    image
  end

  # Mapping to (0,1)
  # score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
  # this lower limit is safe for use with the sigmoid functions and float32
  def sigmoid(data : Slice(Float32)) : Slice(Float32)
    data.map! { |x| 1.0_f32 / (1.0_f32 + Math.exp(-x)) }
  end

  def intersection_over_union(det1 : Detection, det2 : Detection) : Float32
    x_a = [det1.left, det2.left].max
    y_a = [det1.bottom, det2.bottom].max
    x_b = [det1.right, det2.right].min
    y_b = [det1.top, det2.top].min

    inter_area = [x_b - x_a, 0.0_f32].max * [y_b - y_a, 0.0_f32].max

    iou = inter_area / (det1.area + det2.area - inter_area)
    iou
  end

  def non_maximum_suppression(detections : Array(Detection), iou_threshold : Float32) : Array(Detection)
    # Sort detections by score in descending order
    sorted_detections = detections.sort_by! { |d| -d.score }

    suppressed = Array(Bool).new(detections.size, false)
    result = [] of Detection

    sorted_detections.each_with_index do |det, i|
      next if suppressed[i]

      result << det

      sorted_detections.each_with_index do |other_det, j|
        next if suppressed[j]

        if intersection_over_union(det, other_det) > iou_threshold
          suppressed[j] = true
        end
      end
    end

    result
  end

  # This is a trimmed down version of the C++ code;
  # all irrelevant parts have been removed.
  # (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
  # also: https://github.com/patlevin/face-detection-tflite/blob/main/fdlite/face_detection.py#L58
  def generate_anchors(
    strides : Array(Int32),
    anchor_offset_x : Float64 = 0.5,
    anchor_offset_y : Float64 = 0.5,
    aspect_ratios : Array(Float64) = [1.0], # use the aspect ratios your model expects
    scales_per_octave : Int32 = 2,          # default value, adjust if needed
    min_scale : Float64 = 0.1,              # default value, adjust if needed
    max_scale : Float64 = 0.9               # default value, adjust if needed
  ) : Array(Anchor)
    input_size_width, input_size_height = resolution
    anchors = [] of Anchor

    # Generate scales based on scales_per_octave
    octave_scales = (0...scales_per_octave).map { |i| 2.0 ** (i.to_f / scales_per_octave) }

    strides.each do |stride|
      feature_map_height = input_size_height // stride
      feature_map_width = input_size_width // stride

      feature_map_height.times do |y|
        y_center = (y + anchor_offset_y) / feature_map_height.to_f
        feature_map_width.times do |x|
          x_center = (x + anchor_offset_x) / feature_map_width.to_f

          (aspect_ratios.size * octave_scales.size).times { anchors << Anchor.new(x_center.to_f32, y_center.to_f32) }

          # if we are ever interested in calculating the box dimensions
          # aspect_ratios.each do |aspect_ratio|
          #  octave_scales.each do |octave_scale|
          #    # Calculate box width and height for each aspect ratio and scale
          #    box_width = (stride.to_f * octave_scale) * Math.sqrt(aspect_ratio)
          #    box_height = (stride.to_f * octave_scale) / Math.sqrt(aspect_ratio)
          #    anchors << Anchor.new(x_center, y_center, box_width, box_height)
          #  end
          # end
        end
      end
    end

    @anchors = anchors
  end
end
