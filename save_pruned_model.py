import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg
import tensorflow_model_optimization as tfmot


flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_float('initial_sparsity', 0.5, 'initial_sparsity')
flags.DEFINE_float('final_sparsity', 0.8, 'final_sparsity')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
  bbox_tensors = []
  prob_tensors = []
  if FLAGS.tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      elif i == 1:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if FLAGS.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  else:
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
  model.summary()
  
  prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

  # Compute end step to finish pruning after 2 epochs.
  batch_size = 128
  epochs = 2
  validation_split = 0.1 # 10% of training set will be used for validation set. 

  #num_images = train_images.shape[0] * (1 - validation_split)
  #end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
  end_step = 1000

  # # Define model for pruning.
  # pruning_params = {
  #       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=FLAGS.initial_sparsity,
  #                                                                final_sparsity=FLAGS.final_sparsity,
  #                                                                begin_step=0,
  #                                                                end_step=end_step)
  # }

  # model_for_pruning = prune_low_magnitude(model, **pruning_params)

  # # `prune_low_magnitude` requires a recompile.
  # model_for_pruning.compile(optimizer='adam',
  #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #               metrics=['accuracy'])

  # model_for_pruning.summary()
  

  # Helper function uses `prune_low_magnitude` to make only the 
  # Dense layers train with pruning.
  pruning_params = {
         'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=FLAGS.initial_sparsity,
                                                                  final_sparsity=FLAGS.final_sparsity,
                                                                  begin_step=0,
                                                                  end_step=end_step)
  }

  def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
      return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

  # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
  # to the layers of the model.
  model_for_pruning = tf.keras.models.clone_model(
      base_model,
      clone_function=apply_pruning_to_dense,
  )

  model_for_pruning.summary()
 
  model_for_pruning.save(FLAGS.output)

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

