########################################################################

# Code created by Stefan Tippelt
########################################################################

from image_optimize import recursive_optimize
import inception5h
import utils
import os

def process_and_save_img(input_name, category, output_path, image, model, session):
    """
    Function to process and save images to file
    """
    image_properties = {}
    layer_tensors = model.layer_tensors

    # iterate through every other layer
    for layer_tensor in layer_tensors[::2]:
        steps = [x * 0.2 for x in range(0, 5)]
        steps_rounded = [round(x, 2) for x in steps]


        # adjust how much the previous image is blended with the current version
        for blend_number in steps_rounded:
            # just a small speedup to skip the most blended version
            if blend_number == 0.8:
                continue
            else:
                print('blend_number', blend_number)
                img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                                                model=model, session=session,
                                                num_iterations=5, step_size=3.0,
                                                rescale_factor=0.7,
                             num_repeats=3, blend=blend_number)

                # create unique filename in order not to overwrite already created files
                input_name_wo_extension = os.path.splitext(input_name)[0]

                filename = input_name_wo_extension + layer_tensor.name.replace(':', '_') + \
                           str(blend_number).replace('.', '_') + '.jpg'

                # create identifier to split grid into columns
                # grid_identifier = input_name_wo_extension + str(blend_number)

                print('saving image: %s' % filename)
                file = os.path.join(output_path, filename)
                if not os.path.exists(output_path): os.mkdir(output_path)
                utils.save_image(img_result, filename=file)

                # store image properties to dict
                image_properties[filename] = {}
                image_properties[filename]['filename'] = filename
                image_properties[filename]['layer'] = layer_tensor.name
                image_properties[filename]['blend'] = blend_number

    print(image_properties)
    return image_properties


