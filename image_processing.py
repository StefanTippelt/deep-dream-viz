########################################################################

# Code created by Stefan Tippelt
########################################################################

from image_optimize import recursive_optimize
import inception5h
import utils
import os

def process_and_save_img(category, output_path, image, model, session):
    """
    Function to process and save images to file
    """
    image_properties = {}
    layer_tensors = model.layer_tensors

    for layer_tensor in layer_tensors:
        steps = [x * 0.2 for x in range(0, 5)]
        # steps = [x * 0.1 for x in range(0, 10)]
        # adjust how much the previous image is blended with the current version
        for blend_number in steps:
            print('blend_number', blend_number)
            img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                                            model=model, session=session,
                                            num_iterations=5, step_size=3.0,
                                            rescale_factor=0.7,
                         num_repeats=3, blend=blend_number)

            # create unique filename
            filename = category + '_layer_' + layer_tensor.name.replace(':', '_') + \
                       'blend' + str(round(blend_number, 2)).replace('.', '_') + '.jpg'

            print('saving image: %s' % filename)
            file = os.path.join(output_path, filename)
            if not os.path.exists(output_path): os.mkdir(output_path)
            utils.save_image(img_result, filename=file)

            # store image properties to dict
            image_properties[filename] = {}
            image_properties[filename]['layer'] = layer_tensor.name
            image_properties[filename]['blend'] = blend_number
    print(image_properties)
    return image_properties


