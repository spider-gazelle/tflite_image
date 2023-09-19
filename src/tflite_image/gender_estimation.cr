require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image.cr"

class TensorflowLite::Image::GenderEstimation
  include Image::Common

  enum Gender
    Male
    Female
  end

  class Output
    include Detection
    include Detection::Classification

    def initialize(male_score, female_score)
      gender = male_score > female_score ? Gender::Male : Gender::Female
      @score = {male_score, female_score}.max
      @alt_score = {male_score, female_score}.min
      @name = gender.to_s
      @index = gender.value
      @gender = gender
    end

    getter alt_score : Float32
    getter gender : Gender
    getter type : Symbol = :gender
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(image : Canvas) : Tuple(Canvas, Array(Output))
    apply_canvas_to_input_tensor image

    # execute the neural net
    client.invoke!

    # ensure the outputs are a float value between 0 and 1
    outputs = normalize_output_layer client.output(0)

    # transform the results
    detections = [
      Output.new(male_score: outputs[0], female_score: outputs[1]),
    ]

    {image, detections}
  end
end
