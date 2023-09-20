require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image.cr"

class TensorflowLite::Image::Classification
  include Image::Common

  class Output
    include Detection
    include Detection::Classification

    def initialize(@index, @score, @label = nil)
    end

    getter type : Symbol = :class
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(image : Canvas) : Tuple(Canvas, Array(Output))
    apply_canvas_to_input_tensor image

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
      Output.new(
        index: klass,
        label: labels[klass]?,
        score: top_10_values[index]
      )
    end

    {image, detections}
  end
end
