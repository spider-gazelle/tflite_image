# Tensorflow Lite image tools

a library for image classification and feature detection with tflite and crystal lang

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     tflite_image:
       github: spider-gazelle/tflite_image
   ```

2. Run `shards install`

## Usage

Image classification

```crystal
require "tflite_image"

# init the tensorflow client with your classification model
client = TensorflowLite::Client.new("./models/classifier.tflite")

# init the classifier
classifier = TensorflowLite::Image::Classification.new(client)

# load your image
canvas = StumpyJPEG.read("./some_image.jpg")

# run the model, outputs the scaled image that was run through the model
scaled_canvas, detections = classifier.run canvas

# parse the outputs
puts detections.inspect
```

Object detection

```crystal
require "tflite_image"

# init the tensorflow client with your object detection model
client = TensorflowLite::Client.new("./models/classifier.tflite")

# init the detector
classifier = TensorflowLite::Image::ObjectDetection.new(client, scale_mode: :cover)

# load your image
canvas = StumpyJPEG.read("./some_image.jpg")

# run the model, outputs the scaled image that was run through the model
scaled_canvas, detections = classifier.run canvas

# parse the outputs
puts detections.inspect

# markup the image with bounding boxes and save the output:
# ========================================================

# we need to apply offsets to the detections
# as they apply to the scaled_canvas
# so they need adjustment to be mapped back onto the original image
offsets = detector.detection_adjustments(canvas)
detector.markup canvas, detections, *offsets
StumpyPNG.write(canvas, "./bin/detection_output.png")
```

## Contributing

1. Fork it (<https://github.com/spider-gazelle/tflite_image/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Stephen von Takach](https://github.com/stakach) - creator and maintainer
