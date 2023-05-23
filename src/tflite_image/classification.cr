require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image.cr"

class TensorflowLite::Image::Classification
  include Image::Common

  struct Detection
    include JSON::Serializable

    def initialize(@classification, @score, @name)
    end

    getter classification : Int32
    getter score : Float32
    property name : String?
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(canvas : Canvas) : Tuple(Canvas, Array(Detection))
    apply_canvas_to_input_tensor canvas

    # execute the neural net
    client.invoke!

    # ensure the outputs are a float value between 0 and 1
    outputs = normalize_output_layer client.output(0)

    # Sort indices by corresponding value in descending order
    sorted_indices = Array.new(outputs.size) { |i| i }.sort_by! { |i| -outputs[i] }

    # Get top 10 indices and values
    top_10_indices = sorted_indices[0...10]
    top_10_values = top_10_indices.map { |i| outputs[i] }

    # transform the results
    detections = top_10_indices.map_with_index do |klass, index|
      Detection.new(
        classification: klass,
        name: labels[klass]?,
        score: top_10_values[index]
      )
    end

    {canvas, detections}
  end
end
