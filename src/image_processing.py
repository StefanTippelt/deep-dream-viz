########################################################################

# Code created by Stefan Tippelt
########################################################################

import os

from image_optimize import recursive_optimize

import utils


def iteration_layers(model, speedup, session):
    """
    Defines the the layers whose activations are enhanced and therefore
    visualized.

    :param model:
        Inception5h model
    :param speedup:
        speedup selects subset of layers in order to give faster results
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
                         session, speedup=True, step_size=3.0):
    """
    Function to process and save images to file
    step_size default: 3.0
    :param input_name:
    :param category:
    :param output_path:
    :param image:
    :param model:
    :param session:

    """
    if speedup is True:
        num_iterations = 2
    else:
        num_iterations = 5

    image_properties = {}
    layer_tensors = iteration_layers(model, speedup, session)
    print("The following layers will be used for exploration: {}".format(layer_tensors))

    # iterate through defined layer tensors
    for layer_tensor in layer_tensors:
        steps = [x * 0.2 for x in range(0, 5)]
        steps_rounded = [round(x, 2) for x in steps]

        # adjust how much the previous image is blended with current version
        for blend_number in steps_rounded:
            print('blend_number', blend_number)
            img_result = recursive_optimize(layer_tensor=layer_tensor,
                                            image=image,
                                            model=model,
                                            session=session,
                                            num_iterations=num_iterations,
                                            step_size=step_size,
                                            rescale_factor=0.7,
                                            num_repeats=3,
                                            blend=blend_number)

            # create unique filename in order not to overwrite already created
            # files
            input_name_wo_extension = os.path.splitext(input_name)[0]
            filename = input_name_wo_extension + \
                layer_tensor.name.replace(':', '_') + str(blend_number)\
                .replace('.', '_') + '.jpg'

            # create identifier to split grid into columns
            # grid_identifier = input_name_wo_extension + str(blend_number)

            print('saving image: %s' % filename)
            file = os.path.join(output_path, filename)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            utils.save_image(img_result, filename=file)

            # store image properties to dict
            image_properties[filename] = {}
            image_properties[filename]['filename'] = filename
            image_properties[filename]['layer'] = layer_tensor.name
            image_properties[filename]['blend'] = blend_number

    print(image_properties)
    return image_properties
