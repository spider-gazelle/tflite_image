require "tensorflow_lite"
require "stumpy_core"
require "stumpy_resize"
require "../tflite_image.cr"

class TensorflowLite::Image::Segmentation
  include Image::Common

  record Detection, pixels : Slice(Int32), labels : Array(String) do
    include JSON::Serializable

    def labels_detected
      counts = Hash(String, Int32).new { |hash, key| hash[key] = 0 }
      pixels.each do |idx|
        if label = labels[idx]?
          counts[label] += 1
        end
      end
      counts
    end
  end

  # attempts to classify the object, assumes the image has already been prepared
  def process(image : Canvas | FFmpeg::Frame) : Array(Detection)
    apply_canvas_to_input_tensor image

    # execute the neural net
    client.invoke!

    # grab the output details
    output = client.output(0)

    # Note there are multiple versions of segmentation models.
    pixel_labels = case output.rank
                   when 3
                     # each pixel has an integer represenation of the class
                     output.as_type.map &.to_i
                   when 4
                     out_height = output[1]
                     out_width = output[2]
                     out_labels = output[3]

                     # each output is a possible class and we pick the highest index
                     if labels.size == out_labels
                       outputs = normalize_output_layer client.output(0)
                       calculate_labels(outputs, out_height, out_width, out_labels)
                     else
                       # sometimes there is a 1-to-1 matching outputs to pixel components versus
                       # an output per-label as above (input size matches output size)
                       raise NotImplementedError.new("not sure how to interpret the results")
                     end
                   else
                     raise "unknown segmenation model output format: #{output.map(&.to_s).join("x")}"
                   end

    # the scaled image, the label index for each pixel, the list of text labels
    [Detection.new(pixel_labels, labels)]
  end

  protected def calculate_labels(outputs, height, width, label_count)
    # A label per-pixel
    pixel_labels = Slice(Int32).new(width * height)
    pixel_labels.each_index do |index|
      # return a sub-slice of the possible outputs for this pixel
      idx = index * label_count
      results = outputs[idx...(idx + label_count)].map! { |val| val.nan? ? 255.0_f32 : val }

      # find the index of the largest probability for this pixel
      max_index = 0
      max_value = results[0]

      results.each_with_index do |value, output_index|
        if value > max_value
          max_value = value
          max_index = output_index
        end
      end

      # update the label map
      pixel_labels[index] = max_index
    end

    pixel_labels
  end

  # A Colormap for visualizing segmentation results.
  #
  # The colormap is an array with 256 elements, each of which is a 3-element array representing an RGB color.
  class_getter pascal_label_colormap : Array(Tuple(UInt16, UInt16, UInt16)) do
    colormap = Array.new(256) { Array.new(3, 0_u16) }
    indices = Array.new(256, &.to_u16)

    8.times do |shift|
      3.times do |channel|
        256.times do |i|
          colormap[i][channel] |= ((indices[i] >> channel) & 1) << (7 - shift)
        end
      end
      256.times { |i| indices[i] >>= 3 }
    end

    # scale the 8bit colour into the 16bit range
    colormap.map do |colour|
      {
        ((colour[0] / UInt8::MAX) * UInt16::MAX).to_u16,
        ((colour[1] / UInt8::MAX) * UInt16::MAX).to_u16,
        ((colour[2] / UInt8::MAX) * UInt16::MAX).to_u16,
      }
    end
  end

  TRANSPARENT_PIXEL = StumpyCore::RGBA.new(0_u16, 0_u16, 0_u16, 0_u16)
  OPACITY           = UInt16::MAX // 2

  # creates a multi-coloured mask of all the detections in the image
  def build_image_mask(detection : Detection)
    label_map = self.class.pascal_label_colormap
    mask = Canvas.new(*resolution)

    detection.pixels.each_with_index do |label_idx, index|
      # background pixels should be transparent
      if {0, 255}.includes?(label_idx)
        mask.pixels[index] = TRANSPARENT_PIXEL
      else
        mask.pixels[index] = StumpyCore::RGBA.new(*label_map[label_idx], OPACITY)
      end
    end

    mask
  end

  # scales the mask to fit on the image provided
  def scale_image_mask(image : Canvas, mask : Canvas, left_offset : Int32 = 0, top_offset : Int32 = 0, resize_method : StumpyResize::InterpolationMethod = :bilinear)
    scale_image_mask(image.width, image.height, mask, left_offset, top_offset, resize_method)
  end

  # scales the mask to fit the target width and height
  def scale_image_mask(target_width : Int32, target_height : Int32, mask : Canvas, left_offset : Int32 = 0, top_offset : Int32 = 0, resize_method : StumpyResize::InterpolationMethod = :bilinear)
    if left_offset == 0 && top_offset == 0
      return StumpyResize.resize(mask, target_width, target_height, method: resize_method)
    end

    height = target_height - top_offset - top_offset
    width = target_width - left_offset - left_offset
    staging = StumpyResize.resize(mask, width, height, method: resize_method)

    if width > target_width || height > target_height
      # we want to crop the staging image
      StumpyResize.crop(staging, -left_offset, -top_offset, width + left_offset, height + top_offset)
    else
      # we want to pad the staging image
      new_canvas = Canvas.new(target_width, target_height)
      new_canvas.paste(staging, left_offset, top_offset)
      new_canvas
    end
  end
end
