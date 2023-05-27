require "./spec_helper"

SPEC_SEG_IMAGE = Path.new "./bin/seg_image.jpg"

unless File.exists? SPEC_SEG_IMAGE
  puts "downloading image file for segmentation spec..."
  HTTP::Client.get("https://aca.im/downloads/dog_plant.png") do |response|
    raise "could not download segment image file" unless response.success?
    File.write(SPEC_SEG_IMAGE, response.body_io)
  end
end

module TensorflowLite::Image
  describe "segmentation" do
    client = TensorflowLite::Client.new(
      model: URI.parse("https://raw.githubusercontent.com/google-coral/test_data/master/deeplabv3_mnv2_pascal_quant.tflite"),
      labels: URI.parse("https://raw.githubusercontent.com/google-coral/test_data/master/pascal_voc_segmentation_labels.txt")
    )
    seg = Segmentation.new(client)

    it "segments an image" do
      puts client.interpreter.inspect
      puts "input resolution: #{seg.resolution.join("x")}px"
      puts "Labels:\n#{seg.labels}"

      canvas = StumpyPNG.read(SPEC_SEG_IMAGE.expand.to_s)
      scaled_canvas, detections = seg.run canvas
      puts "Unique objects found! pixels: #{detections.pixels.size}, unique: #{detections.labels_detected}"

      mask = seg.build_image_mask detections
      scaled_canvas.paste(mask, 0, 0)

      StumpyPNG.write(mask, "./bin/seg_fit_scaled_mask.png")
      StumpyPNG.write(scaled_canvas, "./bin/seg_fit_scaled_overlay.png")
    end
  end
end
