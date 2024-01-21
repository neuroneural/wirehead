# python imports
import numpy as np

# project imports
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg.labels_to_image_model import labels_to_image_model

# third-party imports
from ext.lab2im import utils, edit_volumes


class BrainGeneratorOptimized:

    def __init__(self,
                 labels_dir,
                 generation_labels=None,
                 n_neutral_labels=None,
                 output_labels=None,
                 subjects_prob=None,
                 batchsize=1,
                 n_channels=1,
                 target_res=None,
                 output_shape=None,
                 output_div_by_n=None,
                 prior_distributions='uniform',
                 generation_classes=None,
                 prior_means=None,
                 prior_stds=None,
                 use_specific_stats_for_channel=False,
                 mix_prior_and_random=False,
                 flipping=True,
                 scaling_bounds=.2,
                 rotation_bounds=15,
                 shearing_bounds=.012,
                 translation_bounds=False,
                 nonlin_std=4.,
                 nonlin_scale=.04,
                 randomise_res=True,
                 max_res_iso=4.,
                 max_res_aniso=8.,
                 data_res=None,
                 thickness=None,
                 bias_field_std=.7,
                 bias_scale=.025,
                 return_gradients=False):
        # prepare data files
        self.labels_paths = utils.list_images_in_folder(labels_dir)
        if subjects_prob is not None:
            self.subjects_prob = np.array(utils.reformat_to_list(subjects_prob, load_as_numpy=True), dtype='float32')
            assert len(self.subjects_prob) == len(self.labels_paths), \
                'subjects_prob should have the same length as labels_path, ' \
                'had {} and {}'.format(len(self.subjects_prob), len(self.labels_paths))
        else:
            self.subjects_prob = None

        # generation parameters
        self.labels_shape, self.aff, self.n_dims, _, self.header, self.atlas_res = \
            utils.get_volume_info(self.labels_paths[0], aff_ref=np.eye(4))
        self.n_channels = n_channels
        if generation_labels is not None:
            self.generation_labels = utils.load_array_if_path(generation_labels)
        else:
            self.generation_labels, _ = utils.get_list_labels(labels_dir=labels_dir)
        if output_labels is not None:
            self.output_labels = utils.load_array_if_path(output_labels)
        else:
            self.output_labels = self.generation_labels
        if n_neutral_labels is not None:
            self.n_neutral_labels = n_neutral_labels
        else:
            self.n_neutral_labels = self.generation_labels.shape[0]
        self.target_res = utils.load_array_if_path(target_res)
        self.batchsize = batchsize
        # preliminary operations
        self.flipping = flipping
        self.output_shape = utils.load_array_if_path(output_shape)
        self.output_div_by_n = output_div_by_n
        # GMM parameters
        self.prior_distributions = prior_distributions
        if generation_classes is not None:
            self.generation_classes = utils.load_array_if_path(generation_classes)
            assert self.generation_classes.shape == self.generation_labels.shape, \
                'if provided, generation_classes should have the same shape as generation_labels'
            unique_classes = np.unique(self.generation_classes)
            assert np.array_equal(unique_classes, np.arange(np.max(unique_classes)+1)), \
                'generation_classes should a linear range between 0 and its maximum value.'
        else:
            self.generation_classes = np.arange(self.generation_labels.shape[0])
        self.prior_means = utils.load_array_if_path(prior_means)
        self.prior_stds = utils.load_array_if_path(prior_stds)
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        # linear transformation parameters
        self.scaling_bounds = utils.load_array_if_path(scaling_bounds)
        self.rotation_bounds = utils.load_array_if_path(rotation_bounds)
        self.shearing_bounds = utils.load_array_if_path(shearing_bounds)
        self.translation_bounds = utils.load_array_if_path(translation_bounds)
        # elastic transformation parameters
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        # blurring parameters
        self.randomise_res = randomise_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.data_res = utils.load_array_if_path(data_res)
        assert not (self.randomise_res & (self.data_res is not None)), \
            'randomise_res and data_res cannot be provided at the same time'
        self.thickness = utils.load_array_if_path(thickness)
        # bias field parameters
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.return_gradients = return_gradients

        # build transformation model
        self.labels_to_image_model, self.model_output_shape = self._build_labels_to_image_model()

        # build generator for model inputs
        self.model_inputs_generator = self._build_model_inputs_generator(mix_prior_and_random)

        # build brain generator
        self.brain_generator = self._build_brain_generator()

    def _build_labels_to_image_model(self):
        # build_model
        lab_to_im_model = labels_to_image_model(labels_shape=self.labels_shape,
                                                n_channels=self.n_channels,
                                                generation_labels=self.generation_labels,
                                                output_labels=self.output_labels,
                                                n_neutral_labels=self.n_neutral_labels,
                                                atlas_res=self.atlas_res,
                                                target_res=self.target_res,
                                                output_shape=self.output_shape,
                                                output_div_by_n=self.output_div_by_n,
                                                flipping=self.flipping,
                                                aff=np.eye(4),
                                                scaling_bounds=self.scaling_bounds,
                                                rotation_bounds=self.rotation_bounds,
                                                shearing_bounds=self.shearing_bounds,
                                                translation_bounds=self.translation_bounds,
                                                nonlin_std=self.nonlin_std,
                                                nonlin_scale=self.nonlin_scale,
                                                randomise_res=self.randomise_res,
                                                max_res_iso=self.max_res_iso,
                                                max_res_aniso=self.max_res_aniso,
                                                data_res=self.data_res,
                                                thickness=self.thickness,
                                                bias_field_std=self.bias_field_std,
                                                bias_scale=self.bias_scale,
                                                return_gradients=self.return_gradients)
        out_shape = lab_to_im_model.output[0].get_shape().as_list()[1:]
        return lab_to_im_model, out_shape

    def _build_model_inputs_generator(self, mix_prior_and_random):
        # build model's inputs generator
        model_inputs_generator = build_model_inputs(path_label_maps=self.labels_paths,
                                                    n_labels=len(self.generation_labels),
                                                    batchsize=self.batchsize,
                                                    n_channels=self.n_channels,
                                                    subjects_prob=self.subjects_prob,
                                                    generation_classes=self.generation_classes,
                                                    prior_means=self.prior_means,
                                                    prior_stds=self.prior_stds,
                                                    prior_distributions=self.prior_distributions,
                                                    use_specific_stats_for_channel=self.use_specific_stats_for_channel,
                                                    mix_prior_and_random=mix_prior_and_random)
        return model_inputs_generator

    def _build_brain_generator(self):
        while True:
            model_inputs = next(self.model_inputs_generator)
            [image, labels] = self.labels_to_image_model.predict(model_inputs)
            yield image, labels

    def generate_brain(self):
        """call this method when an object of this class has been instantiated to generate new brains"""
        return next(self.brain_generator)
