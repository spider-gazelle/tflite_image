require "./classification"
require "./lines"

module TensorflowLite::Image::Detection::BoundingBox
  include Classification
  include Lines

  getter top : Float32
  getter left : Float32
  getter bottom : Float32
  getter right : Float32

  abstract def boundary : Detection::Line

  # Calculate the area
  @[JSON::Field(ignore: true)]
  @[YAML::Field(ignore: true)]
  getter area : Float32 do
    width * height
  end

  # Calculate width and height in percentage
  @[JSON::Field(ignore: true)]
  @[YAML::Field(ignore: true)]
  getter width : Float32 do
    if right >= left
      right - left
    else
      left - right
    end
  end

  @[JSON::Field(ignore: true)]
  @[YAML::Field(ignore: true)]
  getter height : Float32 do
    if bottom >= top
      bottom - top
    else
      top - bottom
    end
  end

  # Calculate current aspect ratio
  @[JSON::Field(ignore: true)]
  @[YAML::Field(ignore: true)]
  getter aspect_ratio : Float64 do
    width.to_f64 / height.to_f64
  end

  # Convert percentage-based dimensions to pixel coordinates
  def to_pixel_coordinates(image_width : Int32, image_height : Int32)
    {
      top:    (image_height * top).to_i,
      left:   (image_width * left).to_i,
      bottom: (image_height * bottom).to_i,
      right:  (image_width * right).to_i,
    }
  end

  # This allows us to extract features from an image in the correct aspect ratio for further processing
  def adjust_bounding_box(target_aspect_ratio : Float64, image_width : Int32, image_height : Int32)
    # Initialize adjustments
    left_adjustment, right_adjustment, top_adjustment, bottom_adjustment = 0_f32, 0_f32, 0_f32, 0_f32

    # Calculate adjustments while maintaining the aspect ratio
    if aspect_ratio == target_aspect_ratio
      # no adjustments, calculate pixels
      return {
        top:    (image_height * top).to_i,
        left:   (image_width * left).to_i,
        bottom: (image_height * bottom).to_i,
        right:  (image_width * right).to_i,
      }
    elsif aspect_ratio < target_aspect_ratio
      # Adjust width
      target_width = height * target_aspect_ratio
      width_diff = target_width - width

      left_adjustment = -width_diff / 2
      right_adjustment = width_diff / 2

      # Check boundaries and adjust
      if (left + left_adjustment) < 0_f32
        overflow = -(left + left_adjustment)
        left_adjustment += overflow
        right_adjustment += overflow # Maintain aspect ratio
      end

      if (right + right_adjustment) > 1_f32
        overflow = right + right_adjustment - 1_f32
        left_adjustment -= overflow
        right_adjustment -= overflow # Maintain aspect ratio
      end
    else
      # Adjust height
      target_height = width / target_aspect_ratio
      height_diff = target_height - height

      top_adjustment = -height_diff / 2
      bottom_adjustment = height_diff / 2

      # Check boundaries and adjust
      if (top + top_adjustment) < 0_f32
        overflow = -(top + top_adjustment)
        top_adjustment += overflow
        bottom_adjustment += overflow # Maintain aspect ratio
      end

      if (bottom + bottom_adjustment) > 1_f32
        overflow = bottom + bottom_adjustment - 1_f32
        top_adjustment -= overflow
        bottom_adjustment -= overflow # Maintain aspect ratio
      end
    end

    # Apply adjustments
    new_left = ({left + left_adjustment, 0_f32}.max * image_width).to_i
    new_right = ({right + right_adjustment, 1_f32}.min * image_width).to_i
    new_top = ({top + top_adjustment, 0_f32}.max * image_height).to_i
    new_bottom = ({bottom + bottom_adjustment, 1_f32}.min * image_height).to_i

    # Convert percentage to pixel coordinates
    {
      top:    {new_top, new_bottom}.min,
      left:   {new_left, new_right}.min,
      bottom: {new_top, new_bottom}.max,
      right:  {new_left, new_right}.max,
    }
  end
end
