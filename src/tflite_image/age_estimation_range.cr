require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image.cr"

class TensorflowLite::Image::AgeEstimationRange
  include Image::Common

  # adjust this for the model you're using
  property ranges = [
    (0..6),
    (7..8),
    (9..11),
    (12..19),
    (20..27),
    (28..35),
    (36..45),
    (46..60),
    (61..75),
  ]

  struct Output
    include Detection
    include Detection::Classification

    def initialize(scores, ranges)
      range = age_estimate(scores, ranges)
      @index = range.begin
      @upper = range.end
      @score = (@index + (@upper - @index)).to_f32 / 2
    end

    getter type : Symbol = :age_range
    getter upper : Int32

    def age_range : Range(Int32, Int32)
      (@index..@upper)
    end

    protected def age_estimate(scores, ranges) : Range(Int32, Int32)
      # Find the maximum value
      max_val = scores.max

      # Find indices close to the max value
      close_indices = [] of Int32
      scores.each_with_index do |val, idx|
        close_indices << idx if (val - max_val).abs <= 0.1_f32
      end

      # calculate the probable range
      if close_indices.size == 1
        ranges[close_indices[0]]
      elsif close_indices.size == 2
        (midpoint(ranges[close_indices[0]])..midpoint(ranges[close_indices[1]]))
      else
        # we'll widen the range a little and estimate older
        (ranges[close_indices[-2]].begin..ranges[close_indices[-1]].end)
      end
    end

    protected def midpoint(range) : Int32
      begins = range.begin
      begins + (range.end - begins) // 2
    end
  end

  # attempts to classify the object, assumes the canvas has already been prepared
  def process(image : Canvas) : Array(Output)
    apply_canvas_to_input_tensor image

    # execute the neural net
    client.invoke!

    # ensure the outputs are a float value between 0 and 1
    outputs = normalize_output_layer client.output(0)

    # transform the results
    detections = [
      Output.new(scores: outputs.to_a, ranges: ranges),
    ]

    detections
  end
end
