import sys
sys.path.append('/data/users1/mdoan4/synth')

from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

# generate an image from the label map.
brain_generator = BrainGenerator('../synthseg/data/training_label_maps/training_seg_01.nii.gz')
im, lab = brain_generator.generate_brain()

# save output image and label map under SynthSeg/generated_examples
utils.save_volume(im, brain_generator.aff, brain_generator.header, './outputs_tutorial_1/image.nii.gz')
utils.save_volume(lab, brain_generator.aff, brain_generator.header, './outputs_tutorial_1/labels.nii.gz')
