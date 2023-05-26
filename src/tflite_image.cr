require "json"
require "stumpy_core"
require "stumpy_resize"

module TensorflowLite::Image
  enum Format
    Float
    Quantized
    QuantizedSigned
  end

  enum Scale
    Fit
    Cover
  end

  DEFAULT_SCALE_MODE = Scale::Fit

  module Common
    def initialize(client : Client, labels : Array(String)? = nil, @scaling_mode : Scale = DEFAULT_SCALE_MODE, input_format : Format? = nil, output_format : Format? = nil)
      @client = client

      # determine format (uint8, int8, float32)
      @input_format = if input_format
                        input_format
                      else
                        # inspect the input layer
                        input_layer = client[0]
                        case input_layer.type
                        when .float32?
                          Format::Float
                        when .u_int8?
                          Format::Quantized
                        when .int8?
                          Format::QuantizedSigned
                        else
                          raise ArgumentError.new("unexpected tensor input format #{input_layer.type}, supports Float32, UInt8 or Int8\nmodel details #{client.interpreter.inspect}")
                        end
                      end

      @labels = if labels
                  labels
                else
                  @labels = client.labels || [] of String
                end
    end

    # the scaling mode to use when preparing images for analysis
    #
    # * Use fit if you would like the whole image to be processed, however letter boxing means the image is smaller which may effect detections
    # * Use cover if you would like to crop the image, only the middle of the image will be used for detection unless the aspect ratio matches the model input
    property scaling_mode : Scale

    # the labels extracted from the model or provided in the initializer
    getter labels : Array(String)

    # the detected tensor format (can be set manually, but not recommended)
    getter input_format : Format

    # the tensorflow lite client
    getter client : Client

    # returns height x width that the models requires
    getter resolution : Tuple(Int32, Int32) do
      input_tensor = client[0]
      # height, width
      {input_tensor[1], input_tensor[2]}
    end

    # scales the image before invoking the tflite model
    def run(canvas : Canvas, scale_mode : Scale = @scaling_mode, resize_method : StumpyResize::InterpolationMethod = :bilinear)
      desired_height, desired_width = resolution
      scaled = case scale_mode
               in .fit?
                 StumpyResize.scale_to_fit(canvas, desired_width, desired_height, resize_method)
               in .cover?
                 StumpyResize.scale_to_cover(canvas, desired_width, desired_height, resize_method)
               end
      process scaled
    end

    # this will calculate the adjustments required to the detections for
    # overlaying on the original image (or a scaled image in the same aspect ratio)
    def detection_adjustments(image : Canvas, scale_mode : Scale = @scaling_mode)
      detection_adjustments(image.width, image.height, scale_mode)
    end

    # :ditto:
    def detection_adjustments(image_width : Int32, image_height : Int32, scale_mode : Scale = @scaling_mode)
      target_height, target_width = resolution
      case scale_mode
      in .fit?
        Image.calculate_boxing_offset(image_width, image_height, target_width, target_height)
      in .cover?
        Image.calculate_cropping_offset(image_width, image_height, target_width, target_height)
      end
    end

    # converts a layers outputs to Float32 for simplified processing
    protected def normalize_output_layer(output_layer) : Slice(Float32)
      case output_layer.type
      when .float32?
        output_layer.as_f32
      when .u_int8?
        output_layer.as_u8.map { |result| (result.to_f32 / 255.0_f32) }
      when .int8?
        output_layer.as_i8.map { |result| (result.to_i + 128).to_f32 / 255.0_f32 }
      else
        raise ArgumentError.new("unexpected tensor output format #{output_layer.type}, supports Float32, UInt8 or Int8\nmodel details #{client.interpreter.inspect}")
      end
    end

    # moves the image colour space into the desired range of the model
    protected def apply_canvas_to_input_tensor(canvas : Canvas)
      input_layer = client[0]
      case input_format
      in .quantized?
        # each pixel component of the canvas has a 16bit range
        inputs = input_layer.as_u8
        canvas.pixels.each_with_index do |rgb, index|
          idx = index * 3
          inputs[idx] = (rgb.r // 256).to_u8
          inputs[idx + 1] = (rgb.g // 256).to_u8
          inputs[idx + 2] = (rgb.b // 256).to_u8
        end
      in .quantized_signed?
        inputs = input_layer.as_i8
        canvas.pixels.each_with_index do |rgb, index|
          idx = index * 3
          inputs[idx] = (rgb.r // 256 - 128).to_i8
          inputs[idx + 1] = (rgb.g // 256 - 128).to_i8
          inputs[idx + 2] = (rgb.b // 256 - 128).to_i8
        end
      in .float?
        inputs = input_layer.as_f32
        canvas.pixels.each_with_index do |rgb, index|
          idx = index * 3
          inputs[idx] = rgb.r.to_f32 / UInt16::MAX
          inputs[idx + 1] = rgb.g.to_f32 / UInt16::MAX
          inputs[idx + 2] = rgb.b.to_f32 / UInt16::MAX
        end
      end
    end
  end

  alias Canvas = StumpyCore::Canvas
end

require "./tflite_image/*"
