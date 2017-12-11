########################################################################
# Code was created on my own, Stefan Tippelt
########################################################################

from image_optimize import recursive_optimize
import inception5h
import utils

def process_and_save_img(image, model, session):
    image_properties = {}
    layer_tensors = model.layer_tensors

    for layer_tensor in layer_tensors:
        steps = [x * 0.2 for x in range(0, 5)]
        # steps = [x * 0.1 for x in range(0, 10)]
        for blend_number in steps:
            print('blend_number', blend_number)
            img_result = recursive_optimize(layer_tensor=layer_tensor, image=image, model=model,
                         session=session, num_iterations=5, step_size=3.0, rescale_factor=0.7,
                         num_repeats=3, blend=blend_number)

            print("Resulting image:")
            utils.plot_image(img_result)

            # TODO join properly
            filename = 'layer_' + layer_tensor.name.replace(':', '_') + 'blend' + str(blend_number).replace('.', '_') + '.jpg'

            print('saving image: %s' % filename)
            utils.save_image(img_result, filename)

            image_properties['filename'] = filename
            image_properties['filename'] = {}
            image_properties['filename']['layer'] = layer_tensor.name
            image_properties['filename']['blend'] = blend_number
            print(image_properties)
    return image_properties


