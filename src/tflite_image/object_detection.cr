require "tensorflow_lite"
require "stumpy_core"
require "stumpy_utils"
require "stumpy_resize"
require "../tflite_image.cr"

class TensorflowLite::Image::ObjectDetection
  include Image::Common

  record Detection, top : Float32, left : Float32, bottom : Float32, right : Float32, classification : Int32, name : String?, score : Float32 do
    include JSON::Serializable

    def lines(width, height, offset_left = 0, offset_top = 0)
      height = height.to_f32 - offset_top - offset_top
      width = width.to_f32 - offset_left - offset_left

      top_px = (top * height).round.to_i + offset_top
      bottom_px = (bottom * height).round.to_i + offset_top
      left_px = (left * width).round.to_i + offset_left
      right_px = (right * width).round.to_i + offset_left

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
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(canvas : Canvas) : Tuple
    {Canvas, Array(Detection)}
    apply_canvas_to_input_tensor canvas

    # execute the neural net
    client.invoke!

    # https://www.tensorflow.org/lite/examples/object_detection/overview#output_signature
    # ensure the outputs are a float value between 0 and 1
    # collate the results (convert bounding boxes from pixels to percentages)
    outputs = client.outputs
    boxes = normalize_output_layer outputs[0]
    features = normalize_output_layer outputs[1]
    scores = normalize_output_layer outputs[2]

    detection_count = normalize_output_layer(outputs[3])[0].to_i

    # transform the results and sort by confidence
    detections = (0...detection_count).map { |index|
      idx = index * 4
      klass = features[index].to_i
      Detection.new(
        top: boxes[idx],
        left: boxes[idx + 1],
        bottom: boxes[idx + 2],
        right: boxes[idx + 3],
        classification: klass,
        name: labels[klass]?,
        score: scores[index]
      )
    }.sort_by! { |d| -d.score }

    {canvas, detections}
  end

  # add the detection details to an image
  #
  # if marking up the original image,
  # you'll need to take into account how it was scaled
  def markup(canvas : Canvas, detections : Array(Detection), left_offset : Int32 = 0, top_offset : Int32 = 0, minimum_score : Float32 = 0.3_f32, font : PCFParser::Font? = nil) : Canvas
    detections.each do |detection|
      next if detection.score < minimum_score

      lines = detection.lines(canvas.width, canvas.height, left_offset, top_offset)
      lines.each do |line|
        canvas.line(*line)
      end

      if font && (label = labels[detection.classification]?)
        canvas.text(lines[0][0], lines[0][1], label, font)
      end
    end
    canvas
  end

  # provide the original image or an image in the same aspect ratio as the original image
  #
  # this will calculate the adjustments required to the detections for
  # overlaying on the original image.
  def detection_adjustments(image : Canvas, scale_mode : Scale = @scaling_mode)
    target_width, target_height = resolution
    case scale_mode
    in .fit?
      calculate_boxing_offset(image.width, image.height, target_width, target_height)
    in .cover?
      calculate_cropping_offset(image.width, image.height, target_width, target_height)
    end
  end

  # how much do we need to adjust detections if we scaled to fit
  def calculate_boxing_offset(original_width : Int32, original_height : Int32, target_width : Int32, target_height : Int32) : Tuple(Int32, Int32)
    # Calculate the scaling factor for width and height
    scale_width = target_width.to_f / original_width.to_f
    scale_height = target_height.to_f / original_height.to_f

    # Decide whether to add letterboxing or pillarboxing
    scale = {scale_width, scale_height}.min

    # Calculate the new scaled dimensions
    scaled_width = (original_width.to_f * scale).round.to_i
    scaled_height = (original_height.to_f * scale).round.to_i

    # Calculate the pixel offset for the left and top in the target image
    left_offset = 0
    top_offset = 0

    if scaled_width < target_width
      # Pillarboxing is required
      left_offset = (target_width - scaled_width) / 2
    end

    if scaled_height < target_height
      # Letterboxing is required
      top_offset = (target_height - scaled_height) / 2
    end

    # Scale the offset back to the original image size
    original_left_offset = (left_offset.to_f / scale).round.to_i
    original_top_offset = (top_offset.to_f / scale).round.to_i

    {-original_left_offset, -original_top_offset}
  end

  # how much we need to adjust detections if we scaled to cover
  def calculate_cropping_offset(original_width : Int32, original_height : Int32, target_width : Int32, target_height : Int32) : Tuple(Int32, Int32)
    # Calculate the scaling factor for width and height
    scale_width = target_width.to_f / original_width.to_f
    scale_height = target_height.to_f / original_height.to_f

    # Decide whether to scale by width or height
    scale = {scale_width, scale_height}.max

    # Calculate the new scaled dimensions
    scaled_width = (original_width.to_f * scale).round.to_i
    scaled_height = (original_height.to_f * scale).round.to_i

    # Calculate the pixel offset for the left and top in the scaled image
    left_offset_scaled = 0
    top_offset_scaled = 0

    if scaled_width > target_width
      # Cropping from the left and right is required
      left_offset_scaled = (scaled_width - target_width) / 2
    end

    if scaled_height > target_height
      # Cropping from the top and bottom is required
      top_offset_scaled = (scaled_height - target_height) / 2
    end

    # Scale the offset back to the original image size
    left_offset = (left_offset_scaled.to_f / scale).round.to_i
    top_offset = (top_offset_scaled.to_f / scale).round.to_i

    {left_offset, top_offset}
  end
end
