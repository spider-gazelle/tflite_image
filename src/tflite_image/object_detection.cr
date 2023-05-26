require "tensorflow_lite"
require "stumpy_core"
require "stumpy_utils"
require "stumpy_resize"
require "../tflite_image.cr"
require "./image_offset_calculations"

class TensorflowLite::Image::ObjectDetection
  include Image::Common

  # top, left, bottom, right are all percentages
  record Detection, top : Float32, left : Float32, bottom : Float32, right : Float32, classification : Int32, name : String?, score : Float32 do
    include JSON::Serializable

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
      return if offset_left == 0 && offset_top == 0

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

    {image, detections}
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

      if font && (label = labels[detection.classification]?)
        image.text(lines[0][0], lines[0][1], label, font)
      end
    end
    image
  end
end
