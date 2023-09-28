module TensorflowLite::Image
  # how much do we need to adjust detections if we scaled to fit
  def self.calculate_boxing_offset(original_width : Int32, original_height : Int32, target_width : Int32, target_height : Int32) : Tuple(Int32, Int32)
    # Calculate the scaling factor for width and height
    scale_width = target_width.to_f / original_width.to_f
    scale_height = target_height.to_f / original_height.to_f

    # Decide whether to add letterboxing or pillarboxing
    scale = {scale_width, scale_height}.min

    # Calculate the new scaled dimensions
    scaled_width = (original_width.to_f * scale).round.to_i
    scaled_height = (original_height.to_f * scale).round.to_i

    # Calculate the pixel offset for the left and top in the target image
    left_offset = 0
    top_offset = 0

    if scaled_width < target_width
      # Pillarboxing is required
      left_offset = (target_width - scaled_width) / 2
    end

    if scaled_height < target_height
      # Letterboxing is required
      top_offset = (target_height - scaled_height) / 2
    end

    # Scale the offset back to the original image size
    original_left_offset = (left_offset.to_f / scale).round.to_i
    original_top_offset = (top_offset.to_f / scale).round.to_i

    {-original_left_offset, -original_top_offset}
  end

  # how much we need to adjust detections if we scaled to cover
  def self.calculate_cropping_offset(original_width : Int32, original_height : Int32, target_width : Int32, target_height : Int32) : Tuple(Int32, Int32)
    # Calculate the scaling factor for width and height
    scale_width = target_width.to_f / original_width.to_f
    scale_height = target_height.to_f / original_height.to_f

    # Decide whether to scale by width or height
    scale = {scale_width, scale_height}.max

    # Calculate the new scaled dimensions
    scaled_width = (original_width.to_f * scale).round.to_i
    scaled_height = (original_height.to_f * scale).round.to_i

    # Calculate the pixel offset for the left and top in the scaled image
    left_offset_scaled = 0
    top_offset_scaled = 0

    if scaled_width > target_width
      # Cropping from the left and right is required
      left_offset_scaled = (scaled_width - target_width) / 2
    end

    if scaled_height > target_height
      # Cropping from the top and bottom is required
      top_offset_scaled = (scaled_height - target_height) / 2
    end

    # Scale the offset back to the original image size
    left_offset = (left_offset_scaled.to_f / scale).round.to_i
    top_offset = (top_offset_scaled.to_f / scale).round.to_i

    {left_offset, top_offset}
  end

  # adjust the detections so they can be applied directly to the source image (or a scaled version in the same aspect ratio)
  #
  # you can run `detection_adjustments` just once and then apply them to detections for each invokation using this function
  def self.adjust(detections : Array, target_width : Int32, target_height : Int32, offset_left : Int32, offset_top : Int32)
    return detections if offset_left.zero? && offset_top.zero?
    height = target_height - offset_top - offset_top
    width = target_width - offset_left - offset_left
    detections.each(&.make_adjustment(width, height, target_width, target_height, offset_left, offset_top))
    detections
  end

  # :ditto:
  def self.adjust(detections : Array, image : Canvas | FFmpeg::Frame, offset_left : Int32, offset_top : Int32)
    adjust(detections, image.width, image.height, offset_left, offset_top)
  end

  # add the detection details to an image
  #
  # if marking up the original image,
  # you'll need to take into account how it was scaled
  def self.markup(image : Canvas, detections : Array, minimum_score : Float32 = 0.3_f32, font : PCFParser::Font? = nil) : Canvas
    detections.each do |detection|
      if detection.responds_to?(:score)
        next if detection.score < minimum_score
      end
      detection.markup(image, minimum_score, font)
    end
    image
  end
end
