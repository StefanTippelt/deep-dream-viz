"""Process, blend and store images.

Code created by Stefan Tippelt
"""

import logging
import os

from image_optimize import recursive_optimize
from PIL import Image, ImageEnhance

import utils

logging.basicConfig(filename='logfile.log', level=logging.INFO)


def iteration_layers(model, speedup, session, indepth_layer=None):
    """
    Define the the layers whose activations are enhanced and visualized.

    Parameters:
        model: Inception5h model
        speedup: selects subset of layers to give faster results
    :return:
        layer tensors to iterate through
    """
    if speedup is True:
        layer_names_reduced = ['conv2d1',
                               'conv2d2',
                               'mixed3b',
                               'mixed4b',
                               'mixed5b']
        layer_tensors = [session.graph.get_tensor_by_name(name + ":0") for name in layer_names_reduced]
    else:
        layer_tensors = model.layer_tensors

    return layer_tensors


def process_and_save_img(input_name, category, output_path, image, model,
                         session, num_repeats, rescale_factor,
                         step_size, speedup=True):
    """
    Function to process and save images to file.

    Parameters
        step_size default: 3.0
        input_name:
        category:
        output_path:
        image:
        model:
        session:
        num_repeats = 3.0:
        rescale_factor = 0.7:
        step_size=3.0:
    return image_properties
    """
    if speedup is True:
        num_iterations = 2
    else:
        num_iterations = 5

    image_properties = {}
    layer_tensors = iteration_layers(model, speedup, session)
    logging.info('The following layers will be used for exploration: %s',
                 layer_tensors)

    # iterate through defined layer tensors
    for layer_tensor in layer_tensors:
        steps = [x * 0.2 for x in range(0, 5)]
        steps_rounded = [round(x, 2) for x in steps]

        # adjust how much the previous image is blended with current version
        for blend_number in steps_rounded:
            # logging.info('blend_number', blend_number)
            img_result = recursive_optimize(layer_tensor=layer_tensor,
                                            image=image,
                                            model=model,
                                            session=session,
                                            num_iterations=num_iterations,
                                            step_size=step_size,
                                            rescale_factor=rescale_factor,
                                            num_repeats=num_repeats,
                                            blend=blend_number)

            # create unique filename in order not to overwrite already created
            # files
            input_name_wo_extension = os.path.splitext(input_name)[0]
            filename = input_name_wo_extension + \
                layer_tensor.name.replace(':', '_') + str(blend_number)\
                .replace('.', '_') + '.jpg'

            # create identifier to split grid into columns
            # grid_identifier = input_name_wo_extension + str(blend_number)

            logging.info('saving image: %s', filename)
            file = os.path.join(output_path, filename)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            utils.save_image(img_result, filename=file)

            # store image properties to dict
            image_properties[filename] = {}
            image_properties[filename]['filename'] = filename
            image_properties[filename]['layer'] = layer_tensor.name
            image_properties[filename]['blend'] = blend_number

    # logging.info('image_properties: ', image_properties)
    return image_properties


def in_depth_layer_preparation(model, session, indepth_layer):
    """
    Get tensor by name for in depth exploration.

    params
    return layer_tensor
    """
    layer_tensor = session.graph.get_tensor_by_name(indepth_layer + ":0")
    return layer_tensor


def process_and_save_img_in_depth(input_name, indepth_layer, category,
                                  output_path, image, model, session,
                                  num_iterations, rescale_factor, num_repeats):
    """
    pass
    """
    image_properties = {}
    layer_tensor = in_depth_layer_preparation(model, session, indepth_layer=indepth_layer)
    logging.info('Layer used for in depth exploration: %s', str(layer_tensor))

    step_sizes = list(range(1, 5))
    steps = [x * 0.2 for x in range(0, 5)]
    steps_rounded = [round(x, 2) for x in steps]

    # adjust how much the previous image is blended with current version
    for step_size in step_sizes:
        logging.info('step size %s', step_size)
        for blend_number in steps_rounded:
            logging.info('blend_number', blend_number)

            img_result = recursive_optimize(layer_tensor=layer_tensor,
                                            image=image,
                                            model=model,
                                            session=session,
                                            num_iterations=num_iterations,
                                            step_size=step_size,
                                            rescale_factor=rescale_factor,
                                            num_repeats=num_repeats,
                                            blend=blend_number)

            # create unique filename in order not to overwrite already created
            # files
            input_name_wo_extension = os.path.splitext(input_name)[0]
            filename = input_name_wo_extension + \
                layer_tensor.name.replace(':', '_') + str(blend_number)\
                .replace('.', '_') + '_step_size_' + str(step_size) + '.jpg'

            # create identifier to split grid into columns
            # grid_identifier = input_name_wo_extension + str(blend_number)

            logging.info('saving image: %s', str(filename))
            file = os.path.join(output_path, filename)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            utils.save_image(img_result, filename=file)

            # store image properties to dict
            image_properties[filename] = {}
            image_properties[filename]['filename'] = filename
            image_properties[filename]['step_size'] = step_size
            image_properties[filename]['blend'] = blend_number

    # logging.info('image properties:', image_properties)
    return image_properties


def resize_secondary_image(primary_image, secondary_image):
    """
    Bring the secondary image to the same size as the primary image.

    Parameters:
        primary_image: image with desired size
        secondary_image: image to be resized
    return resized_secondary_image
    """
    im_primary = Image.open(primary_image)
    im_secondary = Image.open(secondary_image)

    width_primary, height_primary = im_primary.size
    # TODO: add smarter way to do that, take look at resize and crop
    resized_secondary_image = im_secondary.resize((width_primary, height_primary),
                                                  resample=0)
    # resized_secondary_image = resized_secondary_image.convert('LA')

    return resized_secondary_image


def blend_images(primary_image, secondary_image, alpha, saturation_enhance,
                 contrast_enhance):
    """
    Blend two images together.

    After resizing the secondary one to the size of
    the primary one

    Parameters:
        primary_image
        secondary_image
        alpha
        saturation_enhance
        contrast_enhance
    return blended_image
    """
    # TODO: remove colors of blended image
    im_primary = Image.open(primary_image)
    # im_secondary = Image.open(secondary_image)

    resized_secondary_image = resize_secondary_image(primary_image,
                                                     secondary_image)

    # TODO add a smarter way to change color saturation of single images
    saturation = ImageEnhance.Color(resized_secondary_image)
    resized_secondary_image = saturation.enhance(0.0)
    blended_image = Image.blend(im_primary, resized_secondary_image, alpha)

    # TODO: maybe rip that out to be it's own image postprocessing function
    saturation = ImageEnhance.Color(blended_image)
    contrast = ImageEnhance.Contrast(blended_image)

    blended_image = saturation.enhance(saturation_enhance)
    blended_image = contrast.enhance(contrast_enhance)

    return blended_image


# TODO: use this more sophisticated method instead of resize_secondary_image
def resize_and_crop(img_path, modified_path, size, crop_type='top'):
    """
    Resize and crop an image to fit the specified size.

    args:
    img_path: path for the image to resize.
    modified_path: path to store the modified image.
    size: `(width, height)` tuple.
    crop_type: can be 'top', 'middle' or 'bottom', depending on this
    value, the image will cropped getting the 'top/left', 'middle' or
    'bottom/right' of the image to fit the size.
    raises:
    Exception: if can not open the file in img_path of there is problems
    to save the image.
    ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], int(round(size[0] * img.size[1] / img.size[0]))),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, int(round((img.size[1] - size[1]) / 2)), img.size[0],
                int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(round(size[1] * img.size[0] / img.size[1])), size[1]),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (int(round((img.size[0] - size[0]) / 2)), 0,
                int(round((img.size[0] + size[0]) / 2)), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((size[0], size[1]),
            Image.ANTIALIAS)
    # If the scale is the same, we do not need to crop
    img.save(modified_path)
