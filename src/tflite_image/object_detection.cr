require "tensorflow_lite"
require "stumpy_core"
require "stumpy_utils"
require "stumpy_resize"
require "../tflite_image.cr"
require "./image_offset_calculations"

class TensorflowLite::Image::ObjectDetection
  include Image::Common

  # attempts to classify the object, assumes the image has already been prepared
  def process(image : Canvas) : Array(Output)
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
      Output.new(
        top: boxes[idx],
        left: boxes[idx + 1],
        bottom: boxes[idx + 2],
        right: boxes[idx + 3],
        score: scores[index],
        index: klass,
        label: labels[klass]?,
      )
    }.sort_by! { |d| -d.score }

    detections
  end

  # adjust the detections so they can be applied directly to the source image (or a scaled version in the same aspect ratio)
  #
  # you can run `detection_adjustments` just once and then apply them to detections for each invokation using this function
  def adjust(detections : Array(Output), target_width : Int32, target_height : Int32, offset_left : Int32, offset_top : Int32) : Array(Output)
    return detections if offset_left.zero? && offset_top.zero?
    height = target_height - offset_top - offset_top
    width = target_width - offset_left - offset_left
    detections.each(&.make_adjustment(width, height, target_width, target_height, offset_left, offset_top))
    detections
  end

  # :ditto:
  def adjust(detections : Array(Output), image : Canvas, offset_left : Int32, offset_top : Int32) : Array(Output)
    adjust(detections, image.width, image.height, offset_left, offset_top)
  end

  # add the detection details to an image
  #
  # if marking up the original image,
  # you'll need to take into account how it was scaled
  def markup(image : Canvas, detections : Array(Output), minimum_score : Float32 = 0.3_f32, font : PCFParser::Font? = nil) : Canvas
    detections.each do |detection|
      next if detection.score < minimum_score
      detection.markup(image, font)
    end
    image
  end
end

require "./object_detection/*"
