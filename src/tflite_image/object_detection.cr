require "tensorflow_lite"
require "stumpy_core"
require "stumpy_utils"
require "stumpy_resize"
require "../tflite_image.cr"
require "./image_offset_calculations"

class TensorflowLite::Image::ObjectDetection
  include Image::Common

  # attempts to classify the object, assumes the image has already been prepared
  def process(image : Canvas | FFmpeg::Frame) : Array(Output)
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
end

require "./object_detection/*"
