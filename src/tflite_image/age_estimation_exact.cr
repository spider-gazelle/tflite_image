require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image.cr"

class TensorflowLite::Image::AgeEstimationExact
  include Image::Common

  property output_factor : Int32 = 116

  struct Detection
    include JSON::Serializable

    def initialize(@score)
    end

    getter score : Float32
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(image : Canvas) : Tuple(Canvas, Array(Detection))
    apply_canvas_to_input_tensor image

    # execute the neural net
    client.invoke!

    # ensure the outputs are a float value between 0 and 1
    outputs = normalize_output_layer client.output(0)

    # transform the results
    detections = [
      Detection.new(score: outputs[0] * output_factor),
    ]

    {image, detections}
  end
end
